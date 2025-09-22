main_asm.o: main.asm
	nasm -f elf64 main.asm -o main_asm.o -g

main_c.o: main_asm.o
	gcc -c main.c -o main_c.o -g

main: main_asm.o main_c.o
	gcc -o main main_asm.o main_c.o -lm

main.asm:
	python3 compiler.py

clean:
	rm -f *.o
	rm -f main
	rm -f *.asm