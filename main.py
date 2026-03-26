if __name__ != "__main__":

    def print(x):
        _print(str(x))
        _print("\n")


def main() -> None:
    a = ("foo", "bar", "baz")
    b = (69, 420)

    print(a[0])
    print(a[1])
    print(a[2])
    
    print(b[0])
    print(b[1])



main()
