import torch


class MLP(torch.nn.Module):
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden, out_features=out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
