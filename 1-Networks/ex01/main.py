from activations import activate
import torch


def main():
    input = torch.tensor([-2.0000, -0.5000, 0.0000, 0.5000, 2.0000])
    result = activate(input)
    print("Input\t:", input)
    for key, value in result.items():
        print(f"{key}\t: {value}")


if __name__ == "__main__":
    main()
