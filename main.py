if __name__ != "__main__":
    def print(x):
        _print(str(x))
        _print("\n")

def foo(n):
    if n < 5:
        return n + 1
    raise StopIteration

def main() -> None:
    n = 0
    try:
        while 1 > 0:
            n = foo(n)
            print(n)
    except StopIteration:
        print("Exception")
    

main()
