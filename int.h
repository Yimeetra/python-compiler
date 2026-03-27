#pragma once

#include "object.h"
#include <stdint.h>
#include "str.h"
#include "none.h"

typedef struct {
    ObjectHeader header;
    int64_t value;
} IntObj;

StrObj *int__str__(IntObj *self);
IntObj *int__add__(IntObj *self, IntObj *other);
IntObj *int__lt__(IntObj *self, IntObj *other);
IntObj *int__gt__(IntObj *self, IntObj *other);
IntObj *int__le__(IntObj *self, IntObj *other);
IntObj *int__ge__(IntObj *self, IntObj *other);
IntObj *int__eq__(IntObj *self, IntObj *other);
IntObj *int__ne__(IntObj *self, IntObj *other);
void *int__del__(IntObj *self);