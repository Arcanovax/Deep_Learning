import torch
from tensor_ops import operations


def main():
    try:
        result = operations(torch.tensor([[1., 2.],
                                          [3., 4.]]),
                            torch.tensor([[5., 6.],
                                          [7., 8.]]))
    except Exception as e:
        print(e)
    else:
        print(f"Sum:\n{result[0]}")
        print(f"Matmul:\n{result[1]}")
        print(f"Mean of a: {result[2]}")
        print(f"Reshaped a: {result[3]}")

    try:
        result = operations(torch.tensor([[1., 2., 3.],
                                          [4., 5., 6.]]),
                            torch.tensor([[1., 2., 3.],
                                          [4., 5., 6.]]))
    except Exception as e:
        print(e)
    else:
        print(f"Sum:\n{result[0]}")
        print(f"Matmul:\n{result[1]}")
        print(f"Mean of a: {result[2]}")
        print(f"Reshaped a: {result[3]}")


if __name__ == "__main__":
    main()
