#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "none.h"
#include "int.h"

NoneObj *_print(StrObj *string) {
    printf("%s", string->value);
    return None;
}

IntObj *id(void *value) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = (int64_t) value;
    return result;
}

// "__add__"
// "__gt__"
// "__lt__"
// "__ge__"
// "__le__"
// "__eq__"
// "__ne__"
