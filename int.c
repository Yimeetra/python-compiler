#include "int.h"
#include <stdlib.h>
#include <stdio.h>

extern void * None;

StrObj *int__str__(IntObj *self) {
    StrObj *result = malloc(sizeof(StrObj));
    int n = snprintf(0, 0, "%ld", self->value);
    result->value = (char *)malloc(n);
    sprintf(result->value, "%ld", self->value);
    return result;
}

IntObj *int__add__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value + other->value;
    return result;
}

IntObj *int__sub__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value - other->value;
    return result;
}

IntObj *int__mul__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value * other->value;
    return result;
}

IntObj *int__div__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value / other->value;
    return result;
}

IntObj *int__lt__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value < other->value;
    return result;
}

IntObj *int__gt__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value > other->value;
    return result;
}

IntObj *int__le__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value <= other->value;
    return result;
}

IntObj *int__ge__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value >= other->value;
    return result;
}

IntObj *int__eq__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value == other->value;
    return result;
}

IntObj *int__ne__(IntObj *self, IntObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->value != other->value;
    return result;
}

void *int__del__(IntObj *self) {
    free(self);
    return &None;
}