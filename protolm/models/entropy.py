import torch
from torch import nn

from protolm.models.transformer import CausalTransformer


class EntropyModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        h: int,
        d_ff: int,
        n_layers: int,
        dropout: float = 0.0,
        byte_vocab_size: int = 256,
        padding_idx: int = 256,
    ) -> None:
        super().__init__()

        self.padding_idx: int = padding_idx

        self.byte_embedding: nn.Embedding = nn.Embedding(
            num_embeddings=byte_vocab_size + 1,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

        self.transformer: CausalTransformer = CausalTransformer(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            h=h,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.final_norm: nn.LayerNorm = nn.LayerNorm(d_model)

        self.output_head: nn.Linear = nn.Linear(
            in_features=d_model, out_features=byte_vocab_size
        )

    def forward(
        self, byte_ids: torch.Tensor, byte_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            byte_ids: A tensor of byte_ids with shape ``(batch_size, seq_len)`` and dtype ``torch.long``
            byte_mask: A mask with shape ``(batch_size, seq_len)`` and dtype ``torch.bool``

        Result:
            A tensor of raw logits with shape ``(batch_size, seq_len, 256)``
        """
        if byte_mask is None:
            byte_mask = byte_ids != self.padding_idx

        if byte_mask.dtype != torch.bool:
            raise TypeError("byte_mask must have dtype torch.bool")

        if byte_mask.shape != byte_ids.shape:
            raise ValueError(
                f"byte_mask must have the same shape as byte_ids.\nGot {tuple(byte_mask.shape)} and {tuple(byte_ids.shape)}"
            )

        if not byte_mask.any(dim=-1).all().item():
            raise ValueError("each example must contain at least one valid byte.")

        x = self.byte_embedding(byte_ids)  # shape = (batch_size, seq_len, d_model)
        x = self.transformer(x, byte_mask)  # shape = (batch_size, seq_len, d_model)
        x = self.final_norm(x)  # shape = (batch_size, seq_len, d_model)
        logits = self.output_head(x)  # shape = (batch_size, seq_len, byte_vocab_size)

        return logits
