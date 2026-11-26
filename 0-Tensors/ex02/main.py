from autograd import gradient_at


def main():
    x = 3.0
    autograd = gradient_at(x)
    analytical = 6 * x + 2
    print(f"f'({x:g}) computed by autograd: {autograd}")
    print(f"f'({x:g}) analytical (6x+2): {analytical}")
    print(f"Match: {autograd == analytical}")


if __name__ == "__main__":
    main()
