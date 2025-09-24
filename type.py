from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class BuiltinTypesEnum(Enum):
    unknown = auto()

    none = auto()
    int = auto()
    str = auto()

    function = auto()


@dataclass(unsafe_hash=True)
class Type:
    name: str
    payload: Type | None = None
    function_name: str | None = None
    function_arg_types: tuple[Type] = field(default_factory=tuple)

    @staticmethod
    def from_builtin(builtin: BuiltinTypesEnum, **params) -> Type:
        return Type(builtin.name, **params)
