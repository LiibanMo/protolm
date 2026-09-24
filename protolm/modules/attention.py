from typing import override

import torch
from torch import nn

from protolm.modules.embedding import apply_rope, get_rope_frequencies

from ..functional.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, h: int) -> None:
        """Initialize the attention projections.

        Args:
            d_model: Size of each input and output token representation.
            d_k: Size of each head's query and key representation.
            d_v: Size of each head's value representation.
            h: Number of attention heads.
        """
        super().__init__()

        for name, value in {"d_model": d_model, "d_k": d_k, "d_v": d_v, "h": h}.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} is not a positive integer.")

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even. Got {d_k}")

        self.h: int = h
        self.d_model: int = d_model
        self.d_k: int = d_k
        self.d_v: int = d_v

        self.q_proj: nn.Linear = nn.Linear(d_model, h * d_k)
        self.k_proj: nn.Linear = nn.Linear(d_model, h * d_k)
        self.v_proj: nn.Linear = nn.Linear(d_model, h * d_v)

        self.out_proj: nn.Linear = nn.Linear(h * d_v, d_model)

    @override
    def forward(
        self, X: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply projected scaled dot-product attention.

        Args:
            X: Hidden representation of the data after tokenization and positional embedding.
            Has shape ``(batch_size, seq_length, d_model)``
            attn_mask: Optional broadcastable attention mask of shape (seq_length, seq_length)

        Returns:
            The attention output with shape ``(batch_size, seq_len, d_model)``.
        """
        if X.ndim != 3:
            raise ValueError(
                f"X ndim should be 3. If 2, consider unsqueezing. Got {X.ndim}"
            )

        batch_size, seq_len, d_model = X.shape

        if d_model != self.d_model:
            raise ValueError(
                f"X final dimension should match object's d_model attribute. Got {d_model} and {self.d_model}"
            )

        q: torch.Tensor = self.q_proj(X)  # shape = (batch_size, seq_len, h * d_k)
        k: torch.Tensor = self.k_proj(X)  # shape = (batch_size, seq_len, h * d_k)
        v: torch.Tensor = self.v_proj(X)  # shape = (batch_size, seq_len, h * d_v)

        q = torch.reshape(q, (batch_size, seq_len, self.h, self.d_k))
        q = q.transpose(1, 2)  # shape = (batch_size, h, seq_len, d_k)

        k = torch.reshape(k, (batch_size, seq_len, self.h, self.d_k))
        k = k.transpose(1, 2)  # shape = (batch_size, h, seq_len, d_k)

        v = torch.reshape(v, (batch_size, seq_len, self.h, self.d_v))
        v = v.transpose(1, 2)  # shape = (batch_size, h, seq_len, d_v)

        cos, sin = get_rope_frequencies(seq_len, self.d_k, q.dtype, q.device)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn_output = scaled_dot_product_attention(
            q, k, v, attn_mask
        )  # shape = (batch_size, h, seq_len, d_v)

        attn_output = attn_output.transpose(
            1, 2
        )  # shape = (batch_size, seq_len, h, d_v)

        attn_output = torch.reshape(
            attn_output, (batch_size, seq_len, self.h * self.d_v)
        )

        return self.out_proj(attn_output)  # shape = (batch_size, seq, d_model)
