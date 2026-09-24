import torch


def get_rope_frequencies(
    seq_len: int, d: int, dtype: torch.dtype, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.arange(0, d, 2, dtype=dtype, device=device)
    theta = 10_000.0 ** (-i / d)

    positions = torch.arange(seq_len, dtype=dtype, device=device)
    angles = positions[:, None] * theta[None, :]

    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Query OR key tensor with shape ``(batch_size, h, seq_len, d_model)``
        cos: Cosine value with common position and angle of shape ``(seq_len, d_model/2)``
        sin: Sine value with common position and angle of shape ``(seq_len, d_model/2)``

    Returns:
        The position-embedded query/key tensor of original shape.
    """

    x_even = x[..., 0::2]  # (batch_size, h, seq_len, d_model/2)
    x_odd = x[..., 1::2]  # (batch_size, h, seq_len, d_model/2)

    cos = cos[None, None, :, :]  # (1, 1, seq_len, d_model/2)
    sin = sin[None, None, :, :]  # (1, 1, seq_len, d_model/2)

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)
