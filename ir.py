from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from type import Type, BuiltinTypesEnum
from typing import Union


class SourceType(Enum):
    CONST = auto()
    LOCAL = auto()
    GLOBAL = auto()
    TEMP = auto()
    LABEL = auto()


@dataclass(repr=False)
class Source:
    source_type: SourceType
    value: int | str

    def __repr__(self) -> str:
        return f"{self.source_type.name}({self.value})"

    def copy(self) -> Source:
        return Source(self.source_type, self.value)


@dataclass(repr=False)
class TypedSource:
    source: Source
    value_type: Type = Type.from_builtin(BuiltinTypesEnum.unknown)

    def __repr__(self) -> str:
        return f"{self.value_type} {self.source}"

    def copy(self) -> TypedSource:
        return TypedSource(self.source, self.value_type)


@dataclass
class AssignOperation:
    dest: TypedSource
    src: TypedSource


class BinaryOperationEnum(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    EQ = auto()
    NE = auto()


@dataclass
class BinaryOperation:
    binop: BinaryOperationEnum
    dest: TypedSource
    lhs: TypedSource
    rhs: TypedSource


@dataclass
class LabelOperation:
    label: Source


@dataclass
class GotoOperation:
    label: Source


@dataclass
class GotoIfFalseOperation:
    label: Source
    cond: TypedSource


@dataclass
class CallOperation:
    target: Source
    args: list[TypedSource]
    dest: TypedSource


@dataclass
class ReturnOperation:
    value: TypedSource


class _Operation(Enum):
    # ASSIGN = auto()

    # ARG = auto()
    # VA_ARG = auto()
    # CALL = auto()

    BUILD_LIST = auto()
    GET_ITEM = auto()

    # GOTO_IF_FALSE = auto()
    # GOTO = auto()
    # LABEL = auto()

    # RETURN = auto()


Operation = Union[
    AssignOperation,
    BinaryOperation,
    LabelOperation,
    GotoOperation,
    GotoIfFalseOperation,
    CallOperation,
    ReturnOperation,
]


def operation_to_string(op: Operation) -> str:
    match op:
        case AssignOperation(dest, src):
            return f"    {dest} = {src}"
        case BinaryOperation(binop, dest, lhs, rhs):
            return f"    {dest} = {lhs} {binop.name} {rhs}"
        case LabelOperation(label):
            return f"{label.value}:"
        case GotoOperation(label):
            return f"    goto {label.value}"
        case GotoIfFalseOperation(label, cond):
            return f"    goto {label.value} if not {cond}"
        case CallOperation(target, args, dest):
            return f"    call {target.value}({', '.join(map(repr, args))})"
        case ReturnOperation(value):
            return f"    return {value}"
    return repr(op)
