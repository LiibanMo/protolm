import pytest
import torch

from protolm.functional.hashing import roll_poly_hash


@pytest.mark.parametrize("shape", [(5,), (3, 5), (2, 3, 5)])
@pytest.mark.parametrize("buckets,multiplier", [(1, 31), (1000, 31), (1_000_003, 1_000_000_007)])
def test_hash_matches_exact_python_polynomial(shape, buckets, multiplier):
    values = torch.randint(0, 256, shape)
    width = shape[-1]
    # Independent closed-form reference using arbitrary-precision Python ints.
    expected = [
        sum(byte * multiplier ** (width - i - 1) for i, byte in enumerate(row))
        % buckets
        for row in values.reshape(-1, width).tolist()
    ]
    expected = torch.tensor(expected, dtype=torch.long).reshape(shape[:-1])
    actual = roll_poly_hash(values, buckets, multiplier)
    torch.testing.assert_close(actual, expected)
    assert actual.device == values.device
    assert ((actual >= 0) & (actual < buckets)).all()


def test_large_multiplier_precision_regression_and_embedding_lookup():
    actual = roll_poly_hash(torch.tensor([[1, 0]]), 1_000_003)
    torch.testing.assert_close(actual, torch.tensor([997010]))
    buckets = roll_poly_hash(torch.tensor([[1, 2, 3], [3, 2, 1]]), 1000, p=31)
    torch.testing.assert_close(buckets, torch.tensor([26, 946]))
    embedding = torch.nn.Embedding(1000, 4)
    torch.testing.assert_close(embedding(buckets), embedding.weight[[26, 946]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_hash_rejects_non_long_input(dtype):
    with pytest.raises(TypeError, match="torch.long"):
        roll_poly_hash(torch.ones(2, 3, dtype=dtype), 100)


@pytest.mark.parametrize("buckets", [0, -1])
def test_hash_rejects_nonpositive_bucket_count(buckets):
    with pytest.raises(ValueError, match="positive"):
        roll_poly_hash(torch.tensor([[1, 2]]), buckets)


def test_hash_rejects_scalar_input():
    with pytest.raises(ValueError, match="dimension"):
        roll_poly_hash(torch.tensor(1), 100)
