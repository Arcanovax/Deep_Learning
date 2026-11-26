from manual_backward import manual_grads, autograd_grads


def main():
    x = 2.0
    w = 3.0
    b = 1.0
    y = (w * x + b) ** 2
    print(f"x={x} w={w} b={b} -> y={y}")
    manual = manual_grads(x, w, b)
    autograd = autograd_grads(x, w, b)
    print(f"Manual : dy/dw={manual[0]} dy/db={manual[1]}")
    print(f"Autograd : dy/dw={autograd[0]} dy/db={autograd[1]}")
    print(f"Match: {manual == autograd}")


if __name__ == "__main__":
    main()
