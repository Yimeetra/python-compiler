from __future__ import annotations

from dataclasses import dataclass, field
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


class BinaryOperatorEnum(Enum):
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
    binop: BinaryOperatorEnum
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


@dataclass
class GetItemOperation:
    dest: TypedSource
    src: TypedSource
    index: TypedSource


@dataclass
class CommentOperation:
    msg: str


@dataclass
class RaiseOperation:
    handler: int


Operation = Union[
    AssignOperation,
    BinaryOperation,
    LabelOperation,
    GotoOperation,
    GotoIfFalseOperation,
    CallOperation,
    ReturnOperation,
    GetItemOperation,
    CommentOperation,
    RaiseOperation,
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
        case GetItemOperation(dest, src, index):
            return f"    {dest} = {src}[{index}]"
        case CommentOperation(msg):
            return f"    {msg}"
    return repr(op)


op_str_to_op_type: dict[str, BinaryOperatorEnum] = {
    "+": BinaryOperatorEnum.ADD,
    "-": BinaryOperatorEnum.SUB,
    "*": BinaryOperatorEnum.MUL,
    "/": BinaryOperatorEnum.DIV,
    "<": BinaryOperatorEnum.LT,
    ">": BinaryOperatorEnum.GT,
    "<=": BinaryOperatorEnum.LE,
    ">=": BinaryOperatorEnum.GE,
    "==": BinaryOperatorEnum.EQ,
    "!=": BinaryOperatorEnum.NE,
}

op_type_to_method: dict[BinaryOperatorEnum, str] = {
    BinaryOperatorEnum.ADD: "__add__",
    BinaryOperatorEnum.SUB: "__sub__",
    BinaryOperatorEnum.MUL: "__mul__",
    BinaryOperatorEnum.DIV: "__div__",
    BinaryOperatorEnum.LT: "__lt__",
    BinaryOperatorEnum.GT: "__gt__",
    BinaryOperatorEnum.LE: "__le__",
    BinaryOperatorEnum.GE: "__ge__",
    BinaryOperatorEnum.EQ: "__eq__",
    BinaryOperatorEnum.NE: "__ne__",
}


@dataclass
class Frame:
    stack: list[Source] = field(default_factory=list)
    instructions: list[Operation] = field(default_factory=list)
    branches: set[int] = field(default_factory=set)
    exception_handler: int = 0
