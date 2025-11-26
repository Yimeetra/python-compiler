import os

if __name__ == "__main__":
    for file in os.listdir("./tests"):
        if file.endswith(".py"):
            print(file)
            os.system(
                f"python3 compiler.py tests/{file} {file.removesuffix('.py')} tests/build"
            )

            diff_out = os.popen(
                f"./tests/build/{file.removesuffix('.py')} | diff - tests/{file.removesuffix('.py')}_expected"
            ).read()

            if diff_out != "":
                print(f"{file} output differs from expected!")
