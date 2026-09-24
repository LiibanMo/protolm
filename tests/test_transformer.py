import pytest
import torch

from protolm.models.transformer import CausalTransformer, MLP, TransformerBlock


@pytest.fixture(params=["block", "stack"])
def transformer(request):
    arguments = dict(d_model=8, d_k=4, d_v=3, h=2, d_ff=16)
    if request.param == "block":
        return TransformerBlock(**arguments).eval()
    return CausalTransformer(**arguments, n_layers=2).eval()


def test_transformer_shape_and_causality(transformer):
    x = torch.randn(2, 5, 8)
    output = transformer(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    changed = x.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 10
    torch.testing.assert_close(output[:, :3], transformer(changed)[:, :3])
    assert not torch.allclose(output[:, 3:], transformer(changed)[:, 3:])


def test_masked_interior_keys_cannot_influence_later_valid_queries(transformer):
    # An interior mask tests key masking independently of the causal mask.
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, False, True, True], [True, True, False, True]])
    output = transformer(x, mask)
    changed = x.clone()
    changed[~mask] = torch.randn_like(changed[~mask]) * 100
    torch.testing.assert_close(output[mask], transformer(changed, mask)[mask])


def test_zero_residual_branches_preserve_input_exactly():
    block = TransformerBlock(8, 4, 3, 2, 16)
    with torch.no_grad():
        block.mha.out_proj.weight.zero_()
        block.mha.out_proj.bias.zero_()
        block.mlp.linear2.weight.zero_()
        block.mlp.linear2.bias.zero_()
    x = torch.randn(2, 4, 8)
    torch.testing.assert_close(block(x), x, rtol=0, atol=0)


def test_mlp_is_positionwise_and_dropout_respects_mode():
    mlp = MLP(8, 6, 16, dropout=0.5)
    x = torch.randn(2, 4, 8)
    mlp.eval()
    output = mlp(x)
    assert output.shape == (2, 4, 6)
    torch.testing.assert_close(output, mlp(x.reshape(-1, 8)).reshape(2, 4, 6))
    torch.testing.assert_close(output, mlp(x), rtol=0, atol=0)
    mlp.train()
    assert not torch.equal(mlp(x), mlp(x))
