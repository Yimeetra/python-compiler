from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class BuiltinTypesEnum(Enum):
    unknown = auto()
    any = auto()

    none = auto()
    int = auto()
    str = auto()

    list = auto()

    function = auto()


@dataclass(unsafe_hash=True)
class Type:
    name: str
    sub_type: Type | None = None

    @staticmethod
    def from_builtin(builtin: BuiltinTypesEnum, sub_type=None) -> Type:
        return Type(builtin.name, sub_type)

    def __repr__(self):
        return f"{self.name}{f'({self.sub_type})' if self.sub_type else ''}"
