import pytest
import torch

from protolm.device import DeviceInfo


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, True, "cuda"), (True, False, "cuda"),
    (False, True, "mps"), (False, False, "cpu"),
])
def test_device_selection_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.mps, "is_available", lambda: mps)
    assert DeviceInfo.get_device() == expected
