import pytest
import torch

from protolm.modules.attention import MultiHeadAttention


def test_multi_head_attention_applies_forward_mask() -> None:
    attention = MultiHeadAttention(d_model=1, d_k=2, d_v=1, h=1)
    inputs = torch.tensor([[[1.0], [3.0], [5.0]]])
    causal_mask = torch.tril(torch.ones(3, 3, dtype=torch.bool))

    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.q_proj.bias.zero_()
        attention.k_proj.weight.zero_()
        attention.k_proj.bias.zero_()
        attention.v_proj.weight.fill_(1.0)
        attention.v_proj.bias.zero_()
        attention.out_proj.weight.fill_(1.0)
        attention.out_proj.bias.zero_()

    output = attention(inputs, attn_mask=causal_mask)

    expected = torch.tensor([[[1.0], [2.0], [3.0]]])
    torch.testing.assert_close(output, expected)


def test_multiple_heads_concatenate_in_token_order():
    attention = MultiHeadAttention(d_model=6, d_k=2, d_v=3, h=2)
    inputs = torch.randn(2, 4, 6)
    with torch.no_grad():
        attention.v_proj.weight.copy_(torch.eye(6))
        attention.v_proj.bias.zero_()
        attention.out_proj.weight.copy_(torch.eye(6))
        attention.out_proj.bias.zero_()
    # Each query sees only itself: attention must return V, regardless of Q/K.
    output = attention(inputs, torch.eye(4, dtype=torch.bool))
    torch.testing.assert_close(output, inputs)


@pytest.mark.parametrize("name", ["d_model", "d_k", "d_v", "h"])
@pytest.mark.parametrize("value", [0, -2, 1.5])
def test_invalid_attention_dimensions(name, value):
    arguments = dict(d_model=8, d_k=4, d_v=3, h=2)
    arguments[name] = value
    with pytest.raises(ValueError, match=name):
        MultiHeadAttention(**arguments)


def test_odd_rope_dimension_is_rejected():
    with pytest.raises(ValueError, match="d_k must be even"):
        MultiHeadAttention(d_model=8, d_k=3, d_v=2, h=2)


@pytest.mark.parametrize("shape", [(4, 8), (2, 3, 7)])
def test_invalid_hidden_state_shape(shape):
    attention = MultiHeadAttention(d_model=8, d_k=4, d_v=3, h=2)
    with pytest.raises(ValueError):
        attention(torch.zeros(shape))
