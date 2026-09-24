import torch
from torch.nn.utils.rnn import pad_sequence

from protolm._utils import PAD_ID


def pad_batch(
    batch: list[list[int]], value: int = PAD_ID
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        batch: List of bytes of different lengths

    Result:
        A tuple of the padded batch, the key mask and its original row lengths
        (padded_batch, key_mask, lengths)
    """
    if not batch:
        raise ValueError("batch must contain at least one example")

    if any(len(row) == 0 for row in batch):
        raise ValueError("batch must contain at least one byte")

    tensors = [torch.tensor(row, dtype=torch.long) for row in batch]
    lengths = torch.tensor([len(row) for row in batch])

    padded_batch = pad_sequence(tensors, batch_first=True, padding_value=value)

    key_mask = padded_batch != value

    return padded_batch, key_mask, lengths
