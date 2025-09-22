#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void * None;

typedef struct {
    void *__init__;
    void *__le__;
    void *__lt__;
    void *__ne__;
    void *__ge__;
    void *__gt__;
    void *__eq__;
    void *__str__;
} Obj;

typedef struct {
    uint64_t value
} IntObj;

typedef struct {
    char* value
} StrObj;

StrObj *int__str__(IntObj *self) {
    StrObj *result = malloc(sizeof(StrObj));
    result->value = (char *)malloc(128);
    sprintf(result->value, "%zu\n", 69);
    return result;
}

StrObj *str__str__(StrObj *self) {
    return self;
}

void *print(void *string) {
    printf("%s", *(char **)string);
    return None;
}

void print_int(int x) {
    printf("%d", x);
}

void print_string(char *x) {
    printf("%s", x);
}






// "__add__"
// "__gt__"
// "__lt__"
// "__ge__"
// "__le__"
// "__eq__"
// "__ne__"
