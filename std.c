#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include "none.h"
#include "int.h"
#include "type.h"

NoneObj *_print(StrObj *string) {
    printf("%s", string->value);
    return None;
}

IntObj *id(void *value) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = (int64_t) value;
    return result;
}

IntObj *match_exception(TypeObj *exc, ObjectHeader *raised) {
    TypeObj *raised_type;
    IntObj *result = malloc(sizeof(IntObj));
    if (raised->type_id != TYPE_TYPE) {
        raised_type = type(raised);
    } else {
        raised_type = (TypeObj *) raised;
    }


    if (raised_type->type_id == exc->type_id) {

        result->value = 1;
        return result;
    }

    for (int i = 0; i < raised_type->parents_amount; i++) {
        if (raised_type->parent_types[i] == exc->type_id) {
            result->value = 1;
            return result;
        }
    }
    result->value = 0;
    return result;
}

// "__add__"
// "__gt__"
// "__lt__"
// "__ge__"
// "__le__"
// "__eq__"
// "__ne__"
