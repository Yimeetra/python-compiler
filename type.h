#pragma once

#include "object.h"
#include "types.h"
#include "int.h"
#include "str.h"

typedef struct {
    ObjectHeader header;
    uint64_t type_id;
    uint64_t parents_amount;
    uint64_t parent_types[0];
} TypeObj;

TypeObj *type(void *object);
IntObj *type__eq__(TypeObj *self, TypeObj *other);
StrObj *type__str__(TypeObj *self);