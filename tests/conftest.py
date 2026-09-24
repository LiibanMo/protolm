"""Local test imports and reproducible, inexpensive CPU fixtures."""

import sys
from pathlib import Path

import pytest
import torch

# Keep test invocation independent of an editable install, within tests/ only.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session", autouse=True)
def single_threaded_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def reproducible_cpu_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield
