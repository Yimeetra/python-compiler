main_asm.o: main.asm
	nasm -f elf64 main.asm -o main_asm.o -g

main_c.o: main_asm.o
	gcc -c main.c -o main_c.o -g

int.o: int.c
	gcc -c int.c -o int.o -g

str.o: str.c
	gcc -c str.c -o str.o -g

list.o: list.c
	gcc -c list.c -o list.o -g

main: main_asm.o main_c.o int.o str.o list.o
	gcc -o main main_asm.o main_c.o int.o str.o list.o -lm

main.asm:
	python3 compiler.py

clean:
	rm -f *.o
	rm -f main
	rm -f *.asm
	rm -f *.ir