#include "type.h"
#include "types.h"
#include "int.h"
#include "str.h"
#include <stdlib.h>
#include <stdio.h>

TypeObj *type(void *object) {
    TypeObj *result = malloc(sizeof(TypeObj));
    result->header.type_id = TYPE_TYPE;
    result->type = ((ObjectHeader *) object)->type_id;
    return result;
}

IntObj *type__eq__(TypeObj *self, TypeObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = (int64_t) (self->type == other->type);
    return result;
}

StrObj *type__str__(TypeObj *self) {
    StrObj *result = malloc(sizeof(StrObj));
    int n = snprintf(0, 0, "Type(%s)", type_names[self->type]);
    result->value = (char *)malloc(n);
    sprintf(result->value, "Type(%s)", type_names[self->type]);
    return result;
}

typedef struct {
    ObjectHeader header;
} StopIterationObj;

StopIterationObj _StopIteration = {
    .header = TYPE_STOP_ITERATION,
};
StopIterationObj *StopIteration = &_StopIteration;

typedef struct {
    ObjectHeader header;
} ExceptionObj;

ExceptionObj _Exception = {
    .header = TYPE_EXCEPTION,
};
ExceptionObj *Exception = &_Exception;