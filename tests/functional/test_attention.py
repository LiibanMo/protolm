import pytest
import torch
from torch.nn import functional as F

from protolm.functional.attention import scaled_dot_product_attention


def test_causal_mask_broadcasts_over_batch_and_heads() -> None:
    queries = torch.zeros(2, 3, 3, 1)
    keys = torch.zeros(2, 3, 3, 1)
    values = torch.tensor([1.0, 3.0, 5.0]).reshape(1, 1, 3, 1)
    values = values.expand(2, 3, -1, -1)
    causal_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))

    output = scaled_dot_product_attention(
        queries, keys, values, attn_mask=causal_mask
    )

    expected = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1)
    expected = expected.expand_as(output)
    torch.testing.assert_close(output, expected)


def test_attention_rejects_non_boolean_mask() -> None:
    queries = torch.zeros(1, 1, 3, 1)
    keys = torch.zeros(1, 1, 3, 1)
    values = torch.zeros(1, 1, 3, 1)
    mask = torch.ones(3, 3)

    with pytest.raises(TypeError, match="dtype must be torch.bool"):
        scaled_dot_product_attention(queries, keys, values, attn_mask=mask)


def test_attention_rejects_incompatible_mask_shape() -> None:
    queries = torch.zeros(2, 3, 4, 1)
    keys = torch.zeros(2, 3, 4, 1)
    values = torch.zeros(2, 3, 4, 1)
    mask = torch.ones(3, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="is not broadcastable"):
        scaled_dot_product_attention(queries, keys, values, attn_mask=mask)


def test_attention_rejects_mask_on_different_device() -> None:
    queries = torch.zeros(1, 1, 3, 1)
    keys = torch.zeros(1, 1, 3, 1)
    values = torch.zeros(1, 1, 3, 1)
    mask = torch.ones(3, 3, dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="must be on the same device"):
        scaled_dot_product_attention(queries, keys, values, attn_mask=mask)


@pytest.mark.parametrize("masked", [False, True])
def test_attention_matches_pytorch_outputs_and_gradients(masked):
    q = torch.randn(2, 3, 4, 2, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 5, 2, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 5, 3, dtype=torch.float64, requires_grad=True)
    mask = None
    if masked:
        mask = torch.tensor([[True, False, True, False, True],
                             [True, True, False, True, False]])[:, None, None, :]
    actual = scaled_dot_product_attention(q, k, v, mask)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * weight).sum(), (q, k, v), retain_graph=True)
    expected_grads = torch.autograd.grad((expected * weight).sum(), (q, k, v))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_masked_values_have_no_effect_or_gradient():
    q = torch.randn(2, 2, 3, 4)
    k = torch.randn_like(q)
    v = torch.randn(2, 2, 3, 5, requires_grad=True)
    mask = torch.tensor([True, False, True])
    actual = scaled_dot_product_attention(q, k, v, mask)
    changed = v.detach().clone()
    changed[..., 1, :] = 1e6
    torch.testing.assert_close(actual, scaled_dot_product_attention(q, k, changed, mask))
    actual.sum().backward()
    assert torch.count_nonzero(v.grad[..., 1, :]) == 0


@pytest.mark.parametrize("shapes", [
    ((4,), (3, 4), (3, 2)),
    ((3, 4), (4,), (3, 2)),
    ((3, 4), (3, 4), (2,)),
    ((3, 4), (3, 5), (3, 2)),
    ((2, 3, 4), (1, 3, 4), (2, 3, 2)),
    ((3, 4), (3, 4), (2, 2)),
])
def test_attention_rejects_incompatible_qkv(shapes):
    with pytest.raises(ValueError):
        scaled_dot_product_attention(*(torch.zeros(shape) for shape in shapes))


def test_mask_cannot_expand_output_batch_dimensions():
    q = torch.zeros(1, 1, 3, 2)
    with pytest.raises(ValueError, match="Expected"):
        scaled_dot_product_attention(q, q, q, torch.ones(2, 1, 3, 3, dtype=torch.bool))
