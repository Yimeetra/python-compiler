if __name__ != "__main__":
    def print(x):
        _print(str(x))
        _print("\n")


def main() -> None:
    a = type("420, 123")
    b = type("123")
    if a == b:
        print("Yes")
    else:
        print("No")
        
    a = type((420, 123,))
    b = type(123)
    if a == b:
        print("Yes")
    else:
        print("No")
    return None

main()
