if __name__ != "__main__":
    def print(x):
        _print(str(x))
        _print("\n")

def main():
    a = 0

    while a < 10000:
        print("id = " + str(id(a)))
        a += 1



main()
