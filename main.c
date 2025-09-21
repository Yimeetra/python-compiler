#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

extern uint64_t None;

void print(int x) {
    printf("%d", x);
}

void print_int(int x) {
    printf("%d", x);
}

void print_string(char *x) {
    printf("%s", x);
}