import torch

PAD_ID = 256


class DeviceInfo:
    @classmethod
    def get_device(cls) -> str:
        """Select the best available PyTorch compute device.

        Returns:
            The name of the CUDA, MPS, or CPU device to use.
        """
        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"
        else:
            return "cpu"
