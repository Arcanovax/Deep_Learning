

import torch


def operations(a: torch.Tensor, b: torch.Tensor) -> tuple:
    sum = a + b
    try:
        product = a @ b
    except RuntimeError:
        raise AssertionError("Shapes are not compatible for matrix product")

    return (
        sum,
        product,
        a.mean(),
        torch.reshape(a, (1, a.numel()))
    )
