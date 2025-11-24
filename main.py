if __name__ != "__main__":

    def print(x):
        _print(str(x))
        _print("\n")


def main() -> None:
    a = 0
    b = 1
    while a + b < 1000:
        print(a)
        c = a + b
        b = a
        a = c


main()
