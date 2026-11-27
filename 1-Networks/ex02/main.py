
import torch
from train_step import one_step


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


def main():
    model = MLP(4, 8, 3)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    x = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,))

    loss1 = one_step(model, x, y, loss_fn, optimizer)
    print(f"Loss before step 1: {loss1}")
    loss2 = one_step(model, x, y, loss_fn, optimizer)
    print(f"Loss before step 1: {loss2}")

    print(f"Loss decreased: {loss2 < loss1}")


if __name__ == "__main__":
    main()
