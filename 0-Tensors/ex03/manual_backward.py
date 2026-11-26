import torch


def manual_grads(x: float, w: float, b: float):
    partiald_w = 2 * (w*x + b) * x
    partiald_b = 2 * (w*x + b)
    return partiald_w, partiald_b


def autograd_grads(x: float, w: float, b: float):
    tensor_w = torch.tensor(w, requires_grad=True)
    tensor_b = torch.tensor(b, requires_grad=True)

    y = (x * tensor_w + tensor_b) ** 2
    y.backward()

    return tensor_w.grad.item(), tensor_b.grad.item()
