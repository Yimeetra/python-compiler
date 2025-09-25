from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class BuiltinTypesEnum(Enum):
    unknown = auto()
    any = auto()

    none = auto()
    int = auto()
    str = auto()

    function = auto()


@dataclass(unsafe_hash=True)
class Type:
    name: str

    @staticmethod
    def from_builtin(builtin: BuiltinTypesEnum) -> Type:
        return Type(builtin.name)
