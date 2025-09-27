#include "tuple.h"
#include <assert.h>
#include <stdlib.h>

void *tuple__getitem__(TupleObj *self, IntObj *index) {
    assert(index->value < self->length && index->value >= 0); // TODO: currently doesn't support negative indexes
    return self->items[index->value];
}

IntObj *tuple__len__(TupleObj *self) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->length;
    return result;
}