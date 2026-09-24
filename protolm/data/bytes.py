import torch
from torch.utils.data import DataLoader, Dataset

from protolm._utils import PAD_ID
from protolm.data.padding import pad_batch


def encode_text(text: str) -> torch.Tensor:
    encoded = text.encode("utf-8")
    return torch.tensor(list(encoded), dtype=torch.long)


def split_byte_stream(
    byte_ids: torch.Tensor, validation_fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if byte_ids.ndim != 1:
        raise ValueError("byte_ids must be a one-dimensional tensor")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("valudation_fraction must be between 0 and 1")

    if byte_ids.numel() < 4:
        raise ValueError(
            "the byte stream must contain at least four bytes\nso both splits can contain two bytes"
        )

    split_index = int(byte_ids.numel() * (1.0 - validation_fraction))

    split_index = max(2, split_index)
    split_index = min(byte_ids.numel() - 2, split_index)

    return byte_ids[:split_index], byte_ids[split_index:]


def collate_byte_windows(
    examples: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if any(len(example) < 2 for example in examples):
        raise ValueError("each training example must contain at least two bytes")
    return pad_batch(examples, value=PAD_ID)


class ByteWindowDataset(Dataset[list[int]]):
    def __init__(self, byte_ids: torch.Tensor, max_length: int) -> None:
        if byte_ids.ndim != 1:
            raise ValueError("byte_ids must be one-dimensional")

        if byte_ids.dtype != torch.long:
            raise TypeError("byte_ids dtype must be torch.long")

        if max_length < 2:
            raise ValueError("max_length must be at least 2")

        if byte_ids.numel() < 2:
            raise ValueError("the byte stream must contain at least two bytes")

        if not ((0 <= byte_ids) & (byte_ids <= 255)).all().item():
            raise ValueError("byte IDs must be in the range 0 through 255")

        self.byte_ids: torch.Tensor = byte_ids
        self.max_length: int = max_length
        self.stride: int = max_length - 1

        # Stop before the final byte because a one-byte window has no target
        self.starts: list[int] = list(range(0, byte_ids.numel() - 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> list[int]:
        start = self.starts[index]
        end = min(start + self.max_length, self.byte_ids.numel())
        return self.byte_ids[start:end].tolist()


def build_byte_dataloaders(
    text: str,
    *,
    validation_fraction: float,
    max_length: int,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader[list[int]], DataLoader[list[int]]]:
    all_byte_ids = encode_text(text)

    train_byte_ids, val_byte_ids = split_byte_stream(all_byte_ids, validation_fraction)

    train_dataset = ByteWindowDataset(train_byte_ids, max_length)
    val_dataset = ByteWindowDataset(val_byte_ids, max_length)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_byte_windows,
        generator=generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_byte_windows,
        num_workers=0,
    )

    return train_loader, val_loader
