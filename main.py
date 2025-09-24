if __name__ != "__main__":
    def print(x):
        _print(str(x))
        _print("\n")

def main():
    a = 0

    while 1:
        print("id = " + str(id(a)))
        a += 1

    print(a)

main()
