#include "none.h"
#include "types.h"

NoneObj _None = {
    .header = TYPE_NONE
};

NoneObj *None = &_None;