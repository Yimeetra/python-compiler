#include "tuple.h"
#include <assert.h>
#include <stdlib.h>

extern void *None;

void *tuple__getitem__(TupleObj *self, IntObj *index) {
    assert(index->value < self->length && index->value >= 0); // TODO: currently doesn't support negative indexes
    return self->items[index->value];
}

IntObj *tuple__len__(TupleObj *self) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->length;
    return result;
}

typedef struct {
    TupleObj *list;
    int i;
} TupleIterator;

TupleIterator *tuple__iter__(TupleObj *self) {
    TupleIterator *iterator = malloc(sizeof(TupleIterator));
    iterator->list = self;
    iterator->i = 0;
    return iterator; 
}

void *tuple_iterator__next__(TupleIterator *self) {
    if (self->i >= self->list->length) {
        return None;
    }
    IntObj index = {0};
    index.value = self->i;
    void *item = tuple__getitem__(self->list, &index);
    self->i++;
    return item; 
}