import pytest
import torch

from protolm._utils import PAD_ID
from protolm.data.bytes import (
    ByteWindowDataset,
    build_byte_dataloaders,
    collate_byte_windows,
    encode_text,
    split_byte_stream,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", []),
        ("cat", [99, 97, 116]),
        ("Aé", [65, 195, 169]),
        ("🙂", [240, 159, 153, 130]),
    ],
)
def test_encode_text_returns_utf8_byte_ids(text, expected):
    byte_ids = encode_text(text)

    assert byte_ids.dtype == torch.long
    assert byte_ids.ndim == 1
    assert byte_ids.tolist() == expected


def test_split_byte_stream_preserves_order_and_separates_splits():
    byte_ids = torch.arange(10)

    train, validation = split_byte_stream(byte_ids, validation_fraction=0.3)

    torch.testing.assert_close(train, byte_ids[:7])
    torch.testing.assert_close(validation, byte_ids[7:])


@pytest.mark.parametrize("validation_fraction", [0.01, 0.99])
def test_split_byte_stream_keeps_at_least_two_bytes_per_split(validation_fraction):
    train, validation = split_byte_stream(
        torch.arange(6), validation_fraction=validation_fraction
    )

    assert train.numel() >= 2
    assert validation.numel() >= 2
    torch.testing.assert_close(
        torch.cat((train, validation)), torch.arange(6)
    )


@pytest.mark.parametrize("validation_fraction", [0.0, 1.0, -0.1, 1.1])
def test_split_byte_stream_rejects_invalid_fraction(validation_fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_byte_stream(torch.arange(10), validation_fraction)


def test_split_byte_stream_rejects_non_vector_or_insufficient_data():
    with pytest.raises(ValueError, match="one-dimensional"):
        split_byte_stream(torch.arange(6).reshape(2, 3), 0.5)

    with pytest.raises(ValueError, match="at least four bytes"):
        split_byte_stream(torch.arange(3), 0.5)


@pytest.mark.parametrize(
    "byte_count,max_length,expected_windows",
    [
        (2, 4, [[0, 1]]),
        (6, 4, [[0, 1, 2, 3], [3, 4, 5]]),
        (7, 4, [[0, 1, 2, 3], [3, 4, 5, 6]]),
        (8, 4, [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7]]),
    ],
)
def test_byte_windows_overlap_by_one_context_byte(
    byte_count, max_length, expected_windows
):
    dataset = ByteWindowDataset(torch.arange(byte_count), max_length=max_length)

    assert [dataset[index] for index in range(len(dataset))] == expected_windows


@pytest.mark.parametrize("byte_count", range(2, 13))
@pytest.mark.parametrize("max_length", range(2, 7))
def test_byte_windows_cover_each_next_byte_target_once(byte_count, max_length):
    byte_ids = torch.arange(byte_count)
    dataset = ByteWindowDataset(byte_ids, max_length=max_length)

    targets = [target for window in dataset for target in window[1:]]

    assert targets == byte_ids[1:].tolist()
    assert all(2 <= len(window) <= max_length for window in dataset)


@pytest.mark.parametrize(
    "byte_ids,max_length,error_type,message",
    [
        (torch.arange(6).reshape(2, 3), 4, ValueError, "one-dimensional"),
        (torch.arange(4, dtype=torch.float32), 4, TypeError, "torch.long"),
        (torch.tensor([0, 256]), 4, ValueError, "0 through 255"),
        (torch.tensor([-1, 0]), 4, ValueError, "0 through 255"),
        (torch.tensor([0]), 4, ValueError, "at least two bytes"),
        (torch.tensor([0, 1]), 1, ValueError, "at least 2"),
    ],
)
def test_byte_window_dataset_rejects_invalid_inputs(
    byte_ids, max_length, error_type, message
):
    with pytest.raises(error_type, match=message):
        ByteWindowDataset(byte_ids, max_length=max_length)


def test_collate_byte_windows_right_pads_and_returns_lengths():
    byte_ids, byte_mask, lengths = collate_byte_windows(
        [[0, 255, 12], [97, 98]]
    )

    torch.testing.assert_close(
        byte_ids,
        torch.tensor([[0, 255, 12], [97, 98, PAD_ID]]),
    )
    torch.testing.assert_close(
        byte_mask,
        torch.tensor([[True, True, True], [True, True, False]]),
    )
    torch.testing.assert_close(lengths, torch.tensor([3, 2]))
    assert byte_ids.dtype == torch.long
    assert byte_mask.dtype == torch.bool
    assert lengths.dtype == torch.long


@pytest.mark.parametrize("examples", [[], [[1]], [[1, 2], [3]]])
def test_collate_byte_windows_rejects_empty_or_targetless_examples(examples):
    with pytest.raises(ValueError):
        collate_byte_windows(examples)


def test_build_byte_dataloaders_keeps_validation_order_and_batch_contract():
    train_loader, validation_loader = build_byte_dataloaders(
        "abcdefghij",
        validation_fraction=0.3,
        max_length=4,
        batch_size=2,
        seed=42,
    )

    train_batch = next(iter(train_loader))
    validation_batch = next(iter(validation_loader))

    for byte_ids, byte_mask, lengths in (train_batch, validation_batch):
        assert byte_ids.shape == byte_mask.shape
        assert lengths.shape == (byte_ids.shape[0],)
        torch.testing.assert_close(byte_mask.sum(dim=1), lengths)

    validation_ids, validation_mask, validation_lengths = validation_batch
    torch.testing.assert_close(validation_ids, torch.tensor([[104, 105, 106]]))
    assert validation_mask.all()
    torch.testing.assert_close(validation_lengths, torch.tensor([3]))


def test_training_dataloader_shuffle_is_reproducible():
    first_train, _ = build_byte_dataloaders(
        "abcdefghijklmnop",
        validation_fraction=0.25,
        max_length=4,
        batch_size=2,
        seed=7,
    )
    second_train, _ = build_byte_dataloaders(
        "abcdefghijklmnop",
        validation_fraction=0.25,
        max_length=4,
        batch_size=2,
        seed=7,
    )

    first_batches = list(first_train)
    second_batches = list(second_train)

    assert len(first_batches) == len(second_batches)
    for first_batch, second_batch in zip(first_batches, second_batches):
        for first_tensor, second_tensor in zip(first_batch, second_batch):
            torch.testing.assert_close(first_tensor, second_tensor)
