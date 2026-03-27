#pragma once

#include "object.h"
typedef struct {
    ObjectHeader header;
    char* value;
} StrObj;

StrObj *str__str__(StrObj *self);
StrObj *str__add__(StrObj *self, StrObj *other);