#include "list.h"
#include <assert.h>
#include <stdarg.h>

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

void *list__get_item__(ListObj *self, IntObj *index) {
    assert(index->value < self->length && index->value >= 0); // TODO: currently doesn't support negative indexes
    return self->items[index->value];
}