if __name__ != "__main__":
    def print(x):
        _print(str(x))
        _print("\n")


def foo():
    bar()
    
def bar():
    baz()
    
def baz():
    raise

def main() -> None:
    try:
        foo()
    except:
        print("Exception")

main()
