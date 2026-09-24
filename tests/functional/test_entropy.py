import math

import pytest
import torch

from protolm import get_entropy_contributions


@pytest.mark.parametrize(
    "inputs",
    [
        pytest.param(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), id="vector"),
        pytest.param(
            torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float64),
            id="batch",
        ),
        pytest.param(torch.zeros(4, dtype=torch.float64), id="uniform-distribution"),
        pytest.param(torch.tensor([7.0], dtype=torch.float64), id="single-class"),
        pytest.param(
            torch.tensor([0.0, torch.log(torch.tensor(3.0))], dtype=torch.float64),
            id="uneven-binary-distribution",
        ),
        pytest.param(
            torch.tensor([-20.0, 20.0], dtype=torch.float64),
            id="near-deterministic-distribution",
        ),
    ],
)
def test_get_entropy_contributions(inputs: torch.Tensor) -> None:
    expected = torch.distributions.Categorical(logits=inputs).entropy()

    torch.testing.assert_close(get_entropy_contributions(inputs), expected)


def test_get_entropy_contributions_along_selected_dimension() -> None:
    inputs = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
    probabilities = torch.softmax(inputs, dim=0)
    expected = -(probabilities * torch.log_softmax(inputs, dim=0)).sum(dim=0)

    actual = get_entropy_contributions(inputs, dim=0)

    torch.testing.assert_close(actual, expected)


def test_byte_entropy_reduces_vocabulary_not_sequence():
    logits = torch.zeros(2, 5, 256, dtype=torch.float64)
    actual = get_entropy_contributions(logits)
    torch.testing.assert_close(actual, torch.full((2, 5), math.log(256), dtype=logits.dtype))


def test_entropy_is_shift_invariant_and_stable_for_extreme_logits():
    logits = torch.tensor([[[2., 0.], [1000., -1000.]]], dtype=torch.float64)
    actual = get_entropy_contributions(logits)
    torch.testing.assert_close(actual, torch.distributions.Categorical(logits=logits).entropy())
    torch.testing.assert_close(actual, get_entropy_contributions(logits + 10000))
    assert torch.isfinite(actual).all()
    assert actual[0, 1].item() == 0


def test_entropy_gradients_match_categorical_reference():
    logits = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    actual = get_entropy_contributions(logits).sum()
    expected = torch.distributions.Categorical(logits=logits).entropy().sum()
    actual_grad, = torch.autograd.grad(actual, logits, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(actual_grad, expected_grad)
