from __future__ import annotations

import dis
import io
from dataclasses import dataclass, field
from enum import Enum, auto
from types import CodeType

import more_itertools
import itertools

DEBUG = True


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
            case Operation.ASSIGN.name:
                return f"{self.dest} := {self.arg1}"
            case Operation.ARG.name:
                return f"arg {self.arg1}"
            case Operation.CALL.name:
                return f"{self.dest} := call {self.arg1}"
            case Operation.GOTO.name:
                return f"goto {self.dest}"
            case Operation.GOTO_IF_FALSE.name:
                return f"if not {self.arg1} goto {self.dest}"     
            case Operation.RETURN.name:
                return f"ret {self.arg1}"
            case Operation.LABEL.name:
                return f"{self.arg1}: "       
            case _:
                return f"{self.dest} := {self.arg1} {self.op} {self.arg2}"


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


def binop_function_type(arg1: Type, arg2: Type, return_type: Type) -> Type:
    return Type.from_builtin(
        BuiltinTypesEnum.function,
        payload=return_type,
        function_arg_types=(
            arg1,
            arg2
        )
    )

int_methods: dict[str, Type] = {
    "__add__": binop_function_type(
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int)
    ),
    "__str__": Type.from_builtin(
        BuiltinTypesEnum.function,
        payload=Type.from_builtin(BuiltinTypesEnum.str),
        function_arg_types = (
            Type.from_builtin(BuiltinTypesEnum.int)
        )
    ),
}

str_methods: dict[str, Type] = {
    "__add__": binop_function_type(
        Type.from_builtin(BuiltinTypesEnum.str),
        Type.from_builtin(BuiltinTypesEnum.str),
        Type.from_builtin(BuiltinTypesEnum.str)
    ),
    "__str__": Type.from_builtin(
        BuiltinTypesEnum.function,
        payload=Type.from_builtin(BuiltinTypesEnum.str),
        function_arg_types = (
            Type.from_builtin(BuiltinTypesEnum.str)
        )
    ),
}

type_name_to_methods: dict[dict[str, Type]] = {
    "int": int_methods,
    "str": str_methods
}

obj_name_to_type: dict[str, Type] = {
    "int": Type.from_builtin(BuiltinTypesEnum.int),
    "str": Type.from_builtin(BuiltinTypesEnum.str),
}

args_to_regs_map = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]


builtin_functions: dict[str, tuple[str, Type]] = {
    "_print": (
        "_print",
        Type.from_builtin(BuiltinTypesEnum.none)
    ),
    "str": (
        "{arg_types[0].name}__str__",
        Type.from_builtin(BuiltinTypesEnum.str)
    ),
}

class Compiler:
    def __init__(self, input_file_name: str, output_file_name: str) -> None:
        self.input_file_name: str = input_file_name
        self.output_file_name: str = output_file_name
        self._label_generator = self._get_next_label()
        self.output_file = open(output_file_name, "w+")
        self.codes_asm: dict[str, str] = {}
        self.functions: dict[str, CodeType] = {}

    def __del__(self):
        self.output_file.close()

    def _get_next_label(self):
        i = 0
        while 1:
            yield f".L{i}"
            i += 1

    def _generate_function_name(self, base_name: str, arg_types: list[Type]):
        return "_".join([base_name] + [type.name for type in arg_types])

    def generate_constants(self, code_obj: CodeType):
        f = io.StringIO()
        for code in code_obj.co_consts:
            if isinstance(code, CodeType):
                for i, const in enumerate(code.co_consts):
                    f.write(f"{code.co_name}_CONST{i}: \n")
                    match const:
                        case str():
                            f.write(".value_p: dq .value\n")
                            f.write(
                                '.value: db "'
                                + const.replace("\n", '", 0xA, 0xD, "')
                                + '", 0\n'
                            )
                        case int():
                            f.write(f"dq {const}\n")
        f.seek(0)
        return f.read()
    
    def compile_ir(self, code_obj):
        code_obj

        instructions = more_itertools.seekable(dis.get_instructions(code_obj))
        var_iter = itertools.count()
        label_iter = itertools.count()

        jump_labels: dict[str, str] = {}

        for inst in instructions:
            if "JUMP" in inst.opname:
                jump_labels[inst.argval] = f"L{next(label_iter)}"

        stack: list[Source] = []
        output: list[ThreeAddressCode] = []
            
        instructions.seek(0)
        for inst in instructions:
            label = jump_labels.get(inst.offset)
            if label:
                output.append(ThreeAddressCode(
                    Operation.LABEL.name,
                    Source(SourceType.LABEL, label)
                ))
            match inst.opname:
                case "LOAD_CONST":
                    stack.append(Source(SourceType.CONST, inst.arg))
                case "STORE_FAST":
                    output.append(
                        ThreeAddressCode(
                            Operation.ASSIGN.name,
                            stack.pop(),
                            None,
                            Source(SourceType.LOCAL, inst.arg)
                        )
                    )
                case "LOAD_GLOBAL":
                    stack.append(Source(SourceType.GLOBAL, inst.argval))
                case "LOAD_FAST":
                    stack.append(Source(SourceType.LOCAL, inst.arg))
                case "BINARY_OP":
                    var2 = stack.pop()
                    var1 = stack.pop()
                    if "=" in inst.argrepr:
                        op = inst.argrepr.replace("=", "")
                        output.append(ThreeAddressCode(op, var1, var2, var1))
                        stack.append(Source(SourceType.LOCAL, inst.argval))
                    else:
                        temp_var = f"t{next(var_iter)}"
                        output.append(ThreeAddressCode(
                            inst.argrepr,
                            var1, var2,
                            Source(SourceType.TEMP, temp_var)
                        ))
                        stack.append(Source(SourceType.TEMP, temp_var))
                    next(instructions)
                case "CALL":
                    temp_var = f"t{next(var_iter)}"
                    for i in range(inst.argval):
                        output.append(ThreeAddressCode(Operation.ARG.name, stack.pop()))
                    output.append(ThreeAddressCode(
                        Operation.CALL.name,
                        stack.pop(), None,
                        Source(SourceType.TEMP, temp_var)
                    ))
                    stack.append(Source(SourceType.TEMP, temp_var))
                case "COMPARE_OP":
                    var2 = stack.pop()
                    var1 = stack.pop()
                    temp_var = f"t{next(var_iter)}"
                    output.append(ThreeAddressCode(
                        inst.argrepr,
                        var1, var2,
                        Source(SourceType.TEMP, temp_var)
                    ))
                    stack.append(Source(SourceType.TEMP, temp_var))
                case "POP_JUMP_IF_FALSE":
                    var1 = stack.pop()
                    output.append(ThreeAddressCode(
                        Operation.GOTO_IF_FALSE.name,
                        var1, None,
                        Source(SourceType.LABEL, jump_labels[inst.argval])
                    ))
                case "POP_TOP":
                    stack.pop()
                case "JUMP_BACKWARD":
                    output.append(ThreeAddressCode(
                        Operation.GOTO.name,
                        None, None,
                        Source(SourceType.LABEL, jump_labels[inst.argval])
                    ))
                case "RETURN_CONST":
                    output.append(ThreeAddressCode(
                        Operation.RETURN.name,
                        Source(SourceType.CONST, inst.argval
                    )
                    ))
                case "RESUME":
                    pass
                case _:
                    raise Exception(
                        f"Instruction {inst.opname} is unimplemented"
                    )
        return output

    def compile_file(self):
        f = self.output_file
        f.write("BITS 64\n")
        f.write("default rel\n")
        f.write("extern _print\n")
        f.write("extern int__str__\n")
        f.write("extern str__str__\n")
        f.write("extern int__add__\n")
        f.write("extern str__add__\n")

        f.write("section .text\n")

        with open(self.input_file_name, "r", encoding="utf-8") as _f:
            code_obj = compile(_f.read(), self.input_file_name, "exec")

        for code in [
            const for const in code_obj.co_consts if isinstance(const, CodeType)
        ]:
            self.functions[code.co_name] = code

        self.code_to_compile: list[tuple[CodeType, list[Type]]] = []
        self.code_to_compile.append(
            [
                (const, [])
                for const in code_obj.co_consts
                if isinstance(const, CodeType) and const.co_name == "main"
            ][0]
        )

        for code, types in self.code_to_compile:
            if DEBUG:
                print("--------------------------")
                dis.show_code(code)

            ir = self.compile_ir(code)
            name = self.compile_nasm(code, ir, types)

        for name, asm in self.codes_asm.items():
            f.write(asm)

        f.write("section .data\n")

        f.write("global None\n")
        f.write("None: dq 0\n")
        f.write(self.generate_constants(code_obj))

    def compile_nasm(
        self, code_obj: CodeType, ir: list[ThreeAddressCode], arg_types: list[Type] | None = None
    ) -> str:
        variables_types: dict[int, Type] = {}
        temp_type = Type.from_builtin(BuiltinTypesEnum.unknown)

        if arg_types:
            for var, type in zip(
                code_obj.co_varnames,
                itertools.chain(arg_types, itertools.cycle([Type.from_builtin(BuiltinTypesEnum.unknown)]))
            ):
                variables_types[var] = type
        else:
            for var in code_obj.co_varnames:
                variables_types[var] = Type.from_builtin(BuiltinTypesEnum.unknown)

        base_function_name = code_obj.co_name
        function_name: str = self._generate_function_name(
            base_function_name, list(variables_types.values())[0 : code_obj.co_argcount]
        )

        f = io.StringIO()

        f.write(f"global {function_name}\n")
        if DEBUG:
            temp: dict = {
                name: type.name
                for name, type in list(variables_types.items())[: code_obj.co_argcount]
            }
            f.write(f"{function_name}: ; {temp}\n")
        else:
            f.write(f"{function_name}:\n")

        f.write("    push rbp\n")
        f.write("    push rbx\n")
        f.write("    mov rbp, rsp\n")
        f.write(f"    sub rsp, {len(code_obj.co_varnames) * 8}\n")

        for i in range(code_obj.co_argcount):
            f.write(f"    mov [rbp-8*{i + 1}], {args_to_regs_map[i]}\n")

        instructions = more_itertools.seekable(ir)

        last_arg_types: list[Type] = []

        for inst in instructions:
            match inst.op:
                case Operation.ASSIGN.name:
                    var_type = Type.from_builtin(BuiltinTypesEnum.unknown)
                    match inst.arg1.type:
                        case SourceType.CONST:
                            f.write(f"    lea rax, [{base_function_name}_CONST{inst.arg1.value}]\n")
                            obj_name = code_obj.co_consts[inst.arg1.value].__class__.__name__
                            var_type = obj_name_to_type[obj_name]
                        case SourceType.LOCAL:
                            f.write(f"    mov rax, [rbp-{(inst.arg1.value + 1) * 8}]\n")
                            var_name = code_obj.co_varnames[inst.arg1.value]
                            var_type = variables_types[var_name]
                    f.write(f"    mov [rbp-{(inst.dest.value + 1) * 8}], rax\n")
                    variables_types[code_obj.co_varnames[inst.dest.value]] = var_type
                case Operation.ARG.name: 
                    args: list[Source] = []
                    args.append(inst.arg1)
                    while instructions.peek().op != Operation.CALL.name:
                        args.append(next(ir).arg1)
                    
                    args.reverse()

                    for arg, reg in zip(args, args_to_regs_map):
                        if arg.type == SourceType.LOCAL:
                            f.write(f"    mov {reg}, [rbp-{(arg.value + 1) * 8}]\n")
                            var_name = code_obj.co_varnames[arg.value]
                            var_type = variables_types[var_name]
                        elif arg.type == SourceType.TEMP:
                            f.write(f"    mov {reg}, rax\n")
                            var_type = temp_type
                        else:
                            f.write(f"    lea {reg}, [{base_function_name}_CONST{(arg.value)}]\n")
                            obj_name = code_obj.co_consts[inst.arg1.value].__class__.__name__
                            var_type = obj_name_to_type[obj_name]
                        last_arg_types.append(var_type)
                case Operation.CALL.name:
                    name_template, return_type = builtin_functions.get(
                        inst.arg1.value,
                        (
                            self._generate_function_name(
                                inst.arg1.value,
                                last_arg_types
                            ),
                            Type.from_builtin(BuiltinTypesEnum.unknown)
                        )
                    )
                    call_function_name = name_template.format(arg_types=last_arg_types)
                    f.write(f"    call {call_function_name}\n")
                    if (
                        inst.arg1.value not in builtin_functions.keys()
                        and call_function_name not in self.codes_asm.keys()
                    ):
                        self.code_to_compile.append(
                            (
                                self.functions[inst.arg1.value],
                                last_arg_types
                            )
                        )
                    last_arg_types = []
                case Operation.GOTO.name:
                    raise Exception("Unimplemented")
                case Operation.GOTO_IF_FALSE.name:
                    raise Exception("Unimplemented")
                case Operation.RETURN.name:
                    # TODO
                    f.write(f"    add rsp, {len(code_obj.co_varnames) * 8}\n")
                    f.write("    pop rbx\n")
                    f.write("    pop rbp\n")
                    f.write("    ret\n")
                case Operation.LABEL.name:
                    raise Exception("Unimplemented")
                case _:
                    raise Exception("Unimplemented")
        f.seek(0)
        self.codes_asm[function_name] = f.read()
        return function_name


if __name__ == "__main__":
    with open("main.py", "r", encoding="utf-8") as f:
        code_obj = compile(f.read(), "main.py", "exec")
    dis.dis(code_obj)
    dis.show_code(code_obj)

    compiler = Compiler("main.py", "main.asm")
    compiler.compile_file()
