#pragma once

typedef struct {
    char* value;
} StrObj;

StrObj *str__str__(StrObj *self);
StrObj *str__add__(StrObj *self, StrObj *other);