#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void * None;

typedef struct {
    int64_t value;
} IntObj;

typedef struct {
    char* value;
} StrObj;

StrObj *int__str__(IntObj *self) {
    StrObj *result = malloc(sizeof(StrObj));
    int n = snprintf(NULL, 0, "%ld", self->value);
    result->value = (char *)malloc(n);
    sprintf(result->value, "%ld", self->value);
    return result;
}

IntObj *int__add__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value + other->value;
    return result;
}

StrObj *str__str__(StrObj *self) {
    return self;
}

StrObj *str__add__(StrObj *self, StrObj *other) {
    StrObj *result = malloc(sizeof(StrObj));
    result->value = malloc(strlen(self->value) + strlen(other->value));
    strcat(result->value, self->value);
    strcat(result->value, other->value);
    return result;
}

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
