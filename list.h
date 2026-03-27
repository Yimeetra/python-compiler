#pragma once

#include "object.h"
#include "int.h"

typedef struct {
    ObjectHeader header;
    void **items;
    int length;
} ListObj;

ListObj *build_list(int n, ...);
IntObj *list__len__(ListObj *self);
void *list__getitem__(ListObj *self, IntObj *index);