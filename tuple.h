#pragma once

#include "int.h"

typedef struct {
    void **items;
    int length;
} TupleObj;

void *tuple__getitem__(TupleObj *self, IntObj *index);
IntObj *tuple__len__(TupleObj *self);