import torch
from neuron import neuron


def main():
    x = torch.tensor([1., 2., 3.])
    output = neuron(x)
    print("Output :", output)


if __name__ == "__main__":
    main()
