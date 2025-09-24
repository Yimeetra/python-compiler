#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void * None;

void *_print(void *string) {
    printf("%s", *(char **)string);
    return &None;
}

// "__add__"
// "__gt__"
// "__lt__"
// "__ge__"
// "__le__"
// "__eq__"
// "__ne__"
