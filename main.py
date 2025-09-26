if __name__ != "__main__":

    def print(x):
        _print(str(x))
        _print("\n")


def list__str__(self):
    i = 0
    s = "["
    l = len(self)

    while i < l - 1:
        s += str(self[i]) + ", "
        i += 1
    s += str(self[i]) + "]"

    return s


def main() -> None:
    a = [1, 2]

    print(list__str__(a))


main()
