import torch


def gradient_at(value: float) -> float:
    tensor = torch.tensor(value, requires_grad=True)
    x = 3 * (tensor**2) + 2 * tensor + 1
    x.backward()
    return tensor.grad
