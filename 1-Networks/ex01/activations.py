import torch


def activate(x: torch.Tensor) -> dict:
    return {
        "ReLU": torch.nn.functional.relu(x),
        "Sigmoid": torch.nn.functional.sigmoid(x),
        "Tanh": torch.nn.functional.tanh(x)
    }
