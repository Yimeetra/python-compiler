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
    a = (69, 420, 1, 2, "Hello, world!")

    print(a[0])
    print(a[1])
    print(a[2])
    print(a[3])
    print(a[4])

    b = ["foo", "bar"]
    print(list__str__(b))


main()
