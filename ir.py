from dataclasses import dataclass
from enum import Enum, auto

from type import Type, BuiltinTypesEnum


class SourceType(Enum):
    CONST = auto()
    LOCAL = auto()
    GLOBAL = auto()
    TEMP = auto()
    LABEL = auto()


class Operation(Enum):
    ASSIGN = auto()

    ARG = auto()
    VA_ARG = auto()
    CALL = auto()

    BUILD_LIST = auto()
    GET_ITEM = auto()

    GOTO_IF_FALSE = auto()
    GOTO = auto()
    LABEL = auto()

    RETURN = auto()

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


class Source:
    def __init__(self, type: SourceType, value: str | int = ""):
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"{self.type.name}({self.value})"


@dataclass
class ThreeAddressCode:
    op: Operation
    arg1: Source | None = None
    arg2: Source | None = None
    dest: Source | None = None
    dest_type: Type = Type.from_builtin(BuiltinTypesEnum.unknown)
    arg1_type: Type = Type.from_builtin(BuiltinTypesEnum.unknown)
    arg2_type: Type = Type.from_builtin(BuiltinTypesEnum.unknown)

    def __repr__(self):
        match self.op:
            case Operation.ASSIGN:
                return f"{self.dest_type} {self.dest} := {self.arg1}"
            case Operation.ARG:
                return f"arg {self.dest_type} {self.arg1}"
            case Operation.CALL:
                return f"{self.dest_type} {self.dest} := call {self.arg1}"
            case Operation.GOTO:
                return f"goto {self.dest}"
            case Operation.GOTO_IF_FALSE:
                return f"if not {self.dest_type} {self.arg1} goto {self.dest}"
            case Operation.RETURN:
                return f"ret {self.dest_type} {self.arg1}"
            case Operation.LABEL:
                return f"{self.arg1}: "
            case Operation.VA_ARG:
                return f"va_arg {self.dest_type} {self.arg1}"
            case Operation.BUILD_LIST:
                return f"build_list {self.dest_type}"
            case Operation.GET_ITEM:
                return f"{self.dest_type} {self.dest} := {self.arg1}[{self.arg2}]"
            case _:
                return (
                    f"{self.dest_type} {self.dest} := {self.arg1} {self.op} {self.arg2}"
                )
