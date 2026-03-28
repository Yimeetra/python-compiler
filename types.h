#pragma once

typedef enum {
    TYPE_NONE,
    TYPE_INT,
    TYPE_STR,
    TYPE_LIST,
    TYPE_TUPLE,
    TYPE_TYPE,

    TYPE_EXCEPTION,
    TYPE_STOP_ITERATION,
    
    _TYPE_COUNT
} BuiltinType;

static const char * const type_names[] = {
    [TYPE_INT] = "TYPE_INT",
    [TYPE_STR] = "TYPE_STR",
    [TYPE_LIST] = "TYPE_LIST",
    [TYPE_TUPLE] = "TYPE_TUPLE",
    [TYPE_TYPE] = "TYPE_TYPE",
};