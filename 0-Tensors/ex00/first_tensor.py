import torch


def describe(x: torch.Tensor) -> None:
    print("Values: \n", x)
    print("Shape:", x.size())
    print("Dimensions:", x.ndim)
    print("Dtype:", x.dtype)
    print("Device:", x.device)
