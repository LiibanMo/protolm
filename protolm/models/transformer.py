from typing import override

import torch
from torch import nn

from protolm.modules.attention import MultiHeadAttention


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.linear1: nn.Linear = nn.Linear(in_features, hidden_features)
        self.gelu: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.linear2: nn.Linear = nn.Linear(hidden_features, out_features)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        h: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.mha: MultiHeadAttention = MultiHeadAttention(d_model, d_k, d_v, h)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.mlp: MLP = MLP(d_model, d_model, d_ff, dropout)

    @override
    def forward(
        self, x: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.shape[1]

        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        )

        mask = causal_mask[None, None, :, :]  # (1, 1, seq_len, seq_len)

        if key_mask is not None:
            key_mask = key_mask[:, None, None, :]  # (batch_size, 1, 1, seq_len)
            mask = mask & key_mask

        x = x + self.mha(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        h: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, d_k, d_v, h, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

    @override
    def forward(
        self, x: torch.Tensor, key_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_mask)

        return x
