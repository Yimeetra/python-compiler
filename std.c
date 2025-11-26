#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "int.h"

extern void * None;

void *_print(void *string) {
    printf("%s", *(char **)string);
    return &None;
}

IntObj *id(void *value) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = (int64_t) value;
    return result;
}

// "__add__"
// "__gt__"
// "__lt__"
// "__ge__"
// "__le__"
// "__eq__"
// "__ne__"
