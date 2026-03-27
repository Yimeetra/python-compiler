#pragma once

#include "object.h"

typedef struct {
    ObjectHeader header;
} NoneObj;

extern NoneObj _None;
extern NoneObj *None;