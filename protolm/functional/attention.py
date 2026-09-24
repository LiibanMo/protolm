from math import sqrt

import torch


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape ``(batch_size, num_heads, seq_length, d_k)``
        K: Key tensor of shape ``(batch_size, num_heads, seq_length, d_k)``
        V: Value tensor of shape ``(batch_size, num_heads, seq_length, d_v)``.
        attn_mask: An optional broadcastable mask applied to the score matrix QK^T

    Returns:
        The attended values with shape ``(batch_size, num_heads, seq_length, d_v)``.
    """
    if Q.ndim < 2:
        raise ValueError(f"Q should have ndim > 1. Got {Q.ndim}")
    if K.ndim < 2:
        raise ValueError(f"K should have ndim > 1. Got {K.ndim}")
    if V.ndim < 2:
        raise ValueError(f"V should have ndim > 1. Got {V.ndim}")

    if Q.shape[-1] != K.shape[-1]:
        raise ValueError(
            f"Q and K must have same d_k. Got {Q.shape[-1]} and {K.shape[-1]}"
        )

    # all leading dimensions should match
    if not Q.shape[:-2] == K.shape[:-2] == V.shape[:-2]:
        raise ValueError(
            f"All leading dimensions of Q, K and V should match. Got {tuple(Q.shape[:-2])}, {tuple(K.shape[:-2])} and {tuple(V.shape[:-2])}"
        )

    # sequence length must match
    if K.shape[-2] != V.shape[-2]:
        raise ValueError(
            f"Sequence length must match between K and V. Got {K.shape[-2]} and {V.shape[-2]}"
        )

    d_k = Q.shape[-1]
    score_matrix = Q @ K.mT / sqrt(d_k)

    if attn_mask is not None:
        if attn_mask.dtype != torch.bool:
            raise TypeError(
                f"attn_mask dtype must be torch.bool. Got {attn_mask.dtype}"
            )

        if attn_mask.device != score_matrix.device:
            raise ValueError(
                f"attn_mask and attention scores must be on the same device.\nGot {attn_mask.device} and {score_matrix.device}"
            )

        try:
            broadcast_shape = torch.broadcast_shapes(
                attn_mask.shape, score_matrix.shape
            )
        except RuntimeError as error:
            raise ValueError(
                f"mask shape {tuple(attn_mask.shape)} is not broadcastable to attention scores {tuple(score_matrix.shape)}"
            ) from error

        if broadcast_shape != score_matrix.shape:
            raise ValueError(
                f"mask broadcasts to {broadcast_shape}.\n Expected {score_matrix.shape}"
            )

        score_matrix = score_matrix.masked_fill(~attn_mask, float("-inf"))

    return torch.softmax(score_matrix, dim=-1) @ V
