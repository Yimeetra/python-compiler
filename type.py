from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class BuiltinTypesEnum(Enum):
    unknown = auto()
    any = auto()

    none = auto()
    int = auto()
    str = auto()

    list = auto()
    tuple = auto()

    iterator = auto()
    list_iterator = auto()
    tuple_iterator = auto()

    function = auto()


@dataclass(unsafe_hash=True)
class Type:
    name: str
    sub_types: tuple[Type, ...] = field(default_factory=tuple, compare=False)

    @staticmethod
    def from_builtin(
        builtin: BuiltinTypesEnum, sub_types: tuple[Type, ...] | None = None
    ) -> Type:
        return Type(builtin.name, sub_types or ())

    def __repr__(self):
        return f"{self.name}{self.sub_types if self.sub_types != () else ''}"
