import torch
from first_tensor import describe


def main():
    describe(torch.tensor([[1, 2, 3], [4, 5, 6]]))


if __name__ == "__main__":
    main()
