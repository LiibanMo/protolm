import pytest
import torch

from protolm.data.padding import pad_batch


@pytest.mark.parametrize("sentinel", [256, -1])
def test_padding_preserves_bytes_lengths_and_validity(sentinel):
    rows = [[0, 255, 12], [97], [1, 2]]
    padded, mask, lengths = pad_batch(rows, value=sentinel)
    torch.testing.assert_close(lengths, torch.tensor([3, 1, 2]))
    torch.testing.assert_close(
        mask, torch.tensor([[True, True, True], [True, False, False], [True, True, False]])
    )
    assert padded.dtype == torch.long
    assert (padded[~mask] == sentinel).all()
    for i, row in enumerate(rows):
        assert padded[i, mask[i]].tolist() == row


@pytest.mark.parametrize("rows", [[[0]], [[1, 2], [3, 4]]])
def test_equal_length_inputs_need_no_padding(rows):
    padded, mask, lengths = pad_batch(rows)
    torch.testing.assert_close(padded, torch.tensor(rows))
    assert mask.all()
    assert (lengths == len(rows[0])).all()


@pytest.mark.parametrize("rows", [[], [[]], [[], [1, 2]], [[1], []]])
def test_empty_batch_or_example_is_rejected(rows):
    with pytest.raises(ValueError):
        pad_batch(rows)
