from dataclasses import dataclass
from enum import Enum, auto


class SourceType(Enum):
    CONST = auto()
    LOCAL = auto()
    GLOBAL = auto()
    TEMP = auto()
    LABEL = auto()


class Operation(Enum):
    ASSIGN = auto()
    ARG = auto()
    CALL = auto()
    GOTO_IF_FALSE = auto()
    GOTO = auto()
    RETURN = auto()
    LABEL = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()


class Source:
    def __init__(self, type: SourceType, value: str = ""):
        self.type = type
        self.value = value

    def __repr__(self) -> str:
        return f"{self.type.name}({self.value})"


@dataclass
class ThreeAddressCode:
    op: Operation
    arg1: Source
    arg2: Source | None = None
    dest: Source | None = None

    def __repr__(self):
        match self.op:
            case Operation.ASSIGN:
                return f"{self.dest} := {self.arg1}"
            case Operation.ARG:
                return f"arg {self.arg1}"
            case Operation.CALL:
                return f"{self.dest} := call {self.arg1}"
            case Operation.GOTO:
                return f"goto {self.dest}"
            case Operation.GOTO_IF_FALSE:
                return f"if not {self.arg1} goto {self.dest}"
            case Operation.RETURN:
                return f"ret {self.arg1}"
            case Operation.LABEL:
                return f"{self.arg1}: "
            case _:
                return f"{self.dest} := {self.arg1} {self.op} {self.arg2}"
