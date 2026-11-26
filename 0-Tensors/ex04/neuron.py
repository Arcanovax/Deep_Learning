import torch


def neuron(x: torch.Tensor) -> torch.Tensor:
    print("Input :", x)
    layer = torch.nn.Linear(in_features=3, out_features=1)
    layer.weight.data = torch.tensor([[0.1, 0.2, 0.3]])
    layer.bias.data = (torch.tensor([0.0]))
    print("Weights:", layer.weight.data)
    print("Bias :", layer.bias.data)
    output = torch.sigmoid(layer(x))
    return output.detach()
