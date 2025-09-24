#pragma once

#include <stdint.h>
#include "str.h"

typedef struct {
    int64_t value;
} IntObj;

StrObj *int__str__(IntObj *self);
IntObj *int__add__(IntObj *self, IntObj *other);