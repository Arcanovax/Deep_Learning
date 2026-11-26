import torch


def neuron(x: torch.Tensor) -> torch.Tensor:
    layer = torch.nn.Linear(in_features=3, out_features=1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.1, 0.2, 0.3]]))
        layer.bias.copy_(torch.tensor([0.0]))
    print("Input :", x)
    print("Weights:", layer.weight.data)
    print("Bias :", layer.bias.data)
    with torch.no_grad():
        output = torch.sigmoid(layer(x))
    return output
