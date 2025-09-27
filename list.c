#include "list.h"
#include <assert.h>
#include <stdarg.h>
#include <stdlib.h>

extern void *None;

ListObj *build_list(int n, ...) {
    ListObj *self = malloc(sizeof(ListObj));
    self->length = n;
    self->items = malloc(sizeof(void *) * n);
    va_list args;
    va_start(args, n);
    for (int i = 0; i < n; ++i) {
        self->items[i] = va_arg(args, void *);
    }
    va_end(args);
    return self;
}

IntObj *list__len__(ListObj *self) {
    IntObj *result = malloc(sizeof(IntObj));
    result->value = self->length;
    return result;
}

void *list__getitem__(ListObj *self, IntObj *index) {
    assert(index->value < self->length && index->value >= 0); // TODO: currently doesn't support negative indexes
    return self->items[index->value];
}

typedef struct {
    ListObj *list;
    int i;
} ListIterator;

ListIterator *list__iter__(ListObj *self) {
    ListIterator *iterator = malloc(sizeof(ListIterator));
    iterator->list = self;
    iterator->i = 0;
    return iterator; 
}

void *list_iterator__next__(ListIterator *self) {
    if (self->i >= self->list->length) {
        return None;
    }
    IntObj index = {0};
    index.value = self->i;
    void *item = list__getitem__(self->list, &index);
    self->i++;
    return item; 
}