import pytest
import torch

from protolm.modules.embedding import apply_rope, get_rope_frequencies


@pytest.mark.parametrize("width", [2, 4, 8])
def test_frequencies_match_scalar_reference(width):
    cos, sin = get_rope_frequencies(5, width, torch.float64, torch.device("cpu"))
    angles = torch.tensor(
        [[position / 10000 ** (2 * pair / width) for pair in range(width // 2)]
         for position in range(5)], dtype=torch.float64,
    )
    torch.testing.assert_close(cos, angles.cos())
    torch.testing.assert_close(sin, angles.sin())
    assert cos.device.type == sin.device.type == "cpu"


def test_adjacent_pairs_rotate_in_correct_direction():
    x = torch.tensor([[[[1., 2., 3., 4.], [1., 2., 3., 4.]]]])
    cos = torch.tensor([[1., 1.], [0., 0.]])
    sin = torch.tensor([[0., 0.], [1., 1.]])
    expected = torch.tensor([[[[1., 2., 3., 4.], [-2., 1., -4., 3.]]]])
    torch.testing.assert_close(apply_rope(x, cos, sin), expected)


def test_rope_preserves_norm_and_has_correct_gradients():
    x = torch.randn(2, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    cos, sin = get_rope_frequencies(3, 4, x.dtype, x.device)
    rotated = apply_rope(x, cos, sin)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))
    assert torch.autograd.gradcheck(lambda value: apply_rope(value, cos, sin), (x,))


def test_rotated_dot_product_depends_on_relative_position():
    q = torch.randn(1, 1, 1, 8, dtype=torch.float64).expand(1, 1, 7, 8)
    k = torch.randn_like(q[:, :, :1]).expand_as(q)
    cos, sin = get_rope_frequencies(7, 8, q.dtype)
    q_rot, k_rot = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    first = (q_rot[:, :, 1] * k_rot[:, :, 3]).sum(-1)
    shifted = (q_rot[:, :, 4] * k_rot[:, :, 6]).sum(-1)
    torch.testing.assert_close(first, shifted)
