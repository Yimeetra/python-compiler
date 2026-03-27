#pragma once

#include "object.h"
#include "types.h"
#include "int.h"
#include "str.h"

typedef struct {
    ObjectHeader header;
    int type;
} TypeObj;

TypeObj *type(void *object);
IntObj *type__eq__(TypeObj *self, TypeObj *other);
StrObj *type__str__(TypeObj *self);