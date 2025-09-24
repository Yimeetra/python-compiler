#include "str.h"

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