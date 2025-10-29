# Toy python compiler
Currently it is python bytecode to nasm compiler, that using c runtime.

## Quick Start
```
make main
./main
```
It compiles only hardcoded main.py file :)

## How it works?
Firstly three address code is created from python bytecode.
Then types are annotated to every TAC instruction.
And finally nasm is generated using TAC.
