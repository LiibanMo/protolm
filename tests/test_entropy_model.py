import io

import pytest
import torch
from torch.nn import functional as F

from protolm.data.padding import pad_batch
from protolm.functional.entropy import get_entropy_contributions
from protolm.models.entropy import EntropyModel


@pytest.fixture
def model():
    return EntropyModel(d_model=16, d_k=4, d_v=3, h=2, d_ff=32, n_layers=2)


def test_byte_logits_and_entropy_contract(model):
    ids, mask, _ = pad_batch([[0, 255, 12], [97]])
    logits = model(ids)
    assert logits.shape == (2, 3, 256)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, model(ids, mask))
    entropy = get_entropy_contributions(logits)
    assert entropy.shape == ids.shape
    # A controlled head proves forward returns raw scores, not softmax.
    with torch.no_grad():
        model.output_head.weight.zero_()
        model.output_head.bias.copy_(torch.linspace(-2, 2, 256))
    torch.testing.assert_close(model(ids), model.output_head.bias.expand(2, 3, 256))


def test_entropy_model_causality(model):
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    original = model(ids)
    changed = ids.clone()
    changed[:, 3:] += 20
    updated = model(changed)
    torch.testing.assert_close(original[:, :3], updated[:, :3])
    assert not torch.allclose(original[:, 3:], updated[:, 3:])


def test_right_padded_batch_matches_individual_sequences(model):
    model.eval()
    rows = [[0, 255, 1, 2], [3, 4], [5]]
    ids, mask, lengths = pad_batch(rows)
    batched = model(ids, mask)
    for i, length in enumerate(lengths.tolist()):
        standalone = model(torch.tensor([rows[i]]))
        torch.testing.assert_close(batched[i, :length], standalone[0], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("explicit_mask", [False, True])
def test_model_rejects_all_padding_row_in_mixed_batch(model, explicit_mask):
    ids = torch.tensor([[1, 2], [256, 256]])
    mask = ids != 256 if explicit_mask else None
    with pytest.raises(ValueError, match="at least one valid byte"):
        model(ids, mask)


def test_model_rejects_zero_length_sequences(model):
    with pytest.raises(ValueError, match="at least one valid byte"):
        model(torch.empty(2, 0, dtype=torch.long))


def test_model_rejects_non_boolean_mask(model):
    with pytest.raises(TypeError, match="torch.bool"):
        model(torch.tensor([[1, 2]]), torch.ones(1, 2))


@pytest.mark.parametrize(
    "mask_shape",
    [
        pytest.param((1, 3), id="broadcast-batch"),
        pytest.param((2, 1), id="broadcast-sequence"),
        pytest.param((3,), id="missing-batch-dimension"),
        pytest.param((2, 4), id="wrong-sequence-length"),
        pytest.param((2, 1, 3), id="extra-dimension"),
    ],
)
def test_model_rejects_incorrect_mask_shape(model, mask_shape):
    byte_ids = torch.tensor([[1, 2, 3], [4, 5, 256]])
    byte_mask = torch.ones(mask_shape, dtype=torch.bool)
    with pytest.raises(ValueError, match="same shape as byte_ids"):
        model(byte_ids, byte_mask)


def test_shifted_masked_loss_has_finite_gradients_through_all_components(model):
    # This is a test-side training example, not an existing production loss API.
    ids, mask, _ = pad_batch([[10, 11, 12, 13], [14, 15]])
    logits = model(ids)
    logits.retain_grad()
    valid_targets = mask[:, 1:]
    loss = F.cross_entropy(logits[:, :-1][valid_targets], ids[:, 1:][valid_targets])
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for component in (model.byte_embedding, model.transformer, model.final_norm, model.output_head):
        assert any(torch.count_nonzero(p.grad) > 0 for p in component.parameters())
    assert torch.count_nonzero(model.byte_embedding.weight.grad[256]) == 0
    assert torch.count_nonzero(logits.grad[:, -1]) == 0
    assert torch.count_nonzero(logits.grad[:, :-1][~valid_targets]) == 0


def test_state_dict_serialization_preserves_predictions(model):
    model.eval()
    ids = torch.tensor([[1, 2, 3]])
    expected = model(ids)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = EntropyModel(16, 4, 3, 2, 32, 2).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(ids), expected, rtol=0, atol=0)


def test_tiny_next_byte_problem_can_be_overfit():
    model = EntropyModel(16, 4, 4, 2, 32, 1)
    ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0)

    def loss():
        return F.cross_entropy(model(ids)[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1))

    initial = loss().item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        objective = loss()
        objective.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = loss().item()
        predictions = model(ids)[:, :-1].argmax(-1)
    assert final < 0.1, (initial, final)
    assert final < initial * 0.05
    torch.testing.assert_close(predictions, ids[:, 1:])
