#include "type.h"
#include "types.h"
#include "int.h"
#include "str.h"
#include <stdlib.h>
#include <stdio.h>

TypeObj *type(void *object) {
    TypeObj *result = malloc(sizeof(TypeObj));
    result->header.type_id = TYPE_TYPE;
    result->type_id = ((ObjectHeader *) object)->type_id;
    return result;
}

IntObj *type__eq__(TypeObj *self, TypeObj *other) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = (int64_t) (self->type_id == other->type_id);
    return result;
}

StrObj *type__str__(TypeObj *self) {
    StrObj *result = malloc(sizeof(StrObj));
    int n = snprintf(0, 0, "Type(%s)", type_names[self->type_id]);
    result->value = (char *)malloc(n);
    sprintf(result->value, "Type(%s)", type_names[self->type_id]);
    return result;
}


#define DECLARE_STATIC_TYPE(name, type, parents) \
    struct name##TypeObj { \
        TypeObj type_obj; \
        int parent_types[sizeof((int[])parents) / sizeof(int)]; \
    } _##name##TypeObj = { \
        .type_obj.header = { \
            .type_id = TYPE_TYPE, \
        }, \
        .type_obj.type_id = type, \
        .type_obj.parents_amount = sizeof((int[])parents) / sizeof(int), \
        .parent_types = parents, \
    }; \
    TypeObj *name = (TypeObj *) &_##name##TypeObj;


DECLARE_STATIC_TYPE(Exception, TYPE_EXCEPTION, {});
DECLARE_STATIC_TYPE(StopIteration, TYPE_STOP_ITERATION, { TYPE_EXCEPTION });