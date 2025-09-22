from __future__ import annotations

import dis
import io
from dataclasses import dataclass, field
from enum import Enum, auto
from types import CodeType

import more_itertools

DEBUG = True


class BuiltinTypesEnum(Enum):
    UNKNOWN = auto()

    NONE = auto()
    INT = auto()
    STRING = auto()

    FUNCTION = auto()
    NEXT_FUNCTION = auto()

    RANGE = auto()


@dataclass(unsafe_hash=True)
class Type:
    name: str
    payload: Type | None = None
    function_name: str | None = None
    function_arg_types: tuple[Type] = field(default_factory=tuple)

    def from_builtin(builtin: BuiltinTypesEnum, **params):
        return Type(builtin.name, **params)


int_obj_fields: dict[str, Type] = {
    "__add__": Type.from_builtin(
        BuiltinTypesEnum.FUNCTION, payload=Type.from_builtin(BuiltinTypesEnum.INT)
    ),
    "__gt__": Type.from_builtin(
        BuiltinTypesEnum.FUNCTION, payload=Type.from_builtin(BuiltinTypesEnum.INT)
    ),
    "__lt__": Type.from_builtin(
        BuiltinTypesEnum.FUNCTION, payload=Type.from_builtin(BuiltinTypesEnum.INT)
    ),
    "__ge__": Type.from_builtin(
        BuiltinTypesEnum.FUNCTION, payload=Type.from_builtin(BuiltinTypesEnum.INT)
    ),
    "__le__": Type.from_builtin(
        BuiltinTypesEnum.FUNCTION, payload=Type.from_builtin(BuiltinTypesEnum.INT)
    ),
    "__eq__": Type.from_builtin(
        BuiltinTypesEnum.FUNCTION, payload=Type.from_builtin(BuiltinTypesEnum.INT)
    ),
    "__ne__": Type.from_builtin(
        BuiltinTypesEnum.FUNCTION, payload=Type.from_builtin(BuiltinTypesEnum.INT)
    ),
    "__str__": Type.from_builtin(
        BuiltinTypesEnum.FUNCTION, payload=Type.from_builtin(BuiltinTypesEnum.STRING)
    ),
}

string_obj_fields: dict[str, Type] = {
    "__str__": Type.from_builtin(
        BuiltinTypesEnum.FUNCTION, payload=Type.from_builtin(BuiltinTypesEnum.STRING)
    )
}

cmp_op_to_suffix = {">": "g", "<": "l", ">=": "ge", "<=": "le", "==": "e", "!=": "ne"}

bin_op_to_inst = {"+": "add", "-": "sub", "+=": "add", "-=": "sub", "&": "and"}

args_to_regs_map = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
builtin_functions: dict[str, Type] = {
    "print": Type.from_builtin(BuiltinTypesEnum.NONE),
    "print_string": Type.from_builtin(BuiltinTypesEnum.NONE),
    "int__str__": Type.from_builtin(BuiltinTypesEnum.STRING),
}


class Compiler:
    def __init__(self, input_file_name: str, output_file_name: str) -> None:
        self.input_file_name: str = input_file_name
        self.output_file_name: str = output_file_name
        self.jump_labels: dict[int, str] = {}
        self._label_generator = self._get_next_label()
        self.output_file = open(output_file_name, "w+")
        self.codes_asm: dict[str, str] = {}
        self.types: set = {
            Type(BuiltinTypesEnum(i).value, BuiltinTypesEnum(i).name)
            for i in BuiltinTypesEnum._value2member_map_
        }
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

    def _get_function_future_types(
        self,
        code_obj: CodeType,
        instructions: more_itertools.seekable[dis.Instruction],
        variables_types: dict[str, Type],
    ):
        types_stacks: list[list[Type]] = [[]]
        function_name = code_obj.co_name

        for inst in instructions:
            match inst.opname:
                case "LOAD_CONST":
                    match code_obj.co_consts[inst.arg]:
                        case None:
                            types_stacks[-1].append(
                                Type.from_builtin(BuiltinTypesEnum.NONE)
                            )
                        case str():
                            types_stacks[-1].append(
                                Type.from_builtin(BuiltinTypesEnum.STRING)
                            )
                        case int():
                            types_stacks[-1].append(
                                Type.from_builtin(BuiltinTypesEnum.INT)
                            )
                        case _:
                            raise Exception(
                                f'Constant type "{type(inst.argval)}" is unimplemented'
                            )
                case "STORE_FAST":
                    variables_types[inst.argval] = types_stacks[-1].pop()
                case "LOAD_GLOBAL":
                    types_stacks[-1].append(
                        Type.from_builtin(
                            BuiltinTypesEnum.FUNCTION,
                            payload=builtin_functions.get(
                                inst.argval, Type.from_builtin(BuiltinTypesEnum.UNKNOWN)
                            ),
                            function_name=code_obj.co_names[inst.arg // 2],
                        )
                    )
                case "LOAD_FAST":
                    types_stacks[-1].append(variables_types[inst.argval])
                case "BINARY_OP":
                    types_stacks[-1].pop()
                    types_stacks[-1].pop()
                    types_stacks[-1].append(Type.from_builtin(BuiltinTypesEnum.INT))
                case "CALL":
                    arg_types: list[Type] = []
                    for i in range(inst.arg):
                        arg_types.append(types_stacks[-1].pop())
                    if len(types_stacks[-1]) == 0:
                        return arg_types
                    function_type = types_stacks[-1].pop()
                    _function_name = self._generate_function_name(
                        function_type.function_name, arg_types
                    )
                    types_stacks[-1].append(function_type.payload)
                case "POP_TOP":
                    types_stacks[-1].pop()
                case "COMPARE_OP":
                    pass  # TODO
                case "POP_JUMP_IF_FALSE":
                    types_stacks[-1].pop()

    def generate_constants(self, code_obj: CodeType):
        f = io.StringIO()
        for code in code_obj.co_consts:
            if isinstance(code, CodeType):
                for i, const in enumerate(code.co_consts):
                    match const:
                        case str():
                            f.write(f"{code.co_name}_CONST_STRING{i}: \n")
                            f.write(".value_p: dq .value\n")
                            f.write(
                                '.value: db "'
                                + const.replace("\n", '", 0xA, 0xD, "')
                                + '", 0\n'
                            )
                        case int():
                            f.write(f"{code.co_name}_CONST_INT{i}: ")
                            f.write(f"dq {const}\n")
        f.seek(0)
        return f.read()

    def compile_file(self):
        f = self.output_file
        f.write("BITS 64\n")
        f.write("default rel\n")
        f.write("extern print_int\n")
        f.write("extern print_string\n")
        f.write("extern print\n")
        f.write("extern int__str__\n")

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

            name = self.compile_code(code, types)
            f.write(self.codes_asm[name])

        f.write("section .data\n")

        f.write("global None\n")
        f.write("None: dq 0\n")
        f.write(self.generate_constants(code_obj))

    def compile_code(
        self, code_obj: CodeType, arg_types: list[Type] | None = None
    ) -> str:
        types_stacks: list[list[Type]] = [[]]
        variables_types: dict[str, Type] = {}

        if arg_types:
            for var, type in zip(code_obj.co_varnames, arg_types):
                variables_types[var] = type
        else:
            for var in code_obj.co_varnames:
                variables_types[var] = Type.from_builtin(BuiltinTypesEnum.UNKNOWN)

        function_base_name = code_obj.co_name
        function_name: str = self._generate_function_name(
            function_base_name, list(variables_types.values())[0 : code_obj.co_argcount]
        )

        f = io.StringIO()
        jump_labels = {}
        instructions = more_itertools.seekable(dis.get_instructions(code_obj))
        for inst in instructions:
            if "JUMP" in inst.opname:
                jump_labels[inst.argval] = next(self._label_generator)

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

        instructions.seek(0)
        for inst in instructions:
            label = jump_labels.get(inst.offset)
            if label:
                f.write(f"{label}:\n")
            try:
                if DEBUG:
                    if "JUMP" in inst.opname:
                        f.write(
                            f"; {inst.offset} {inst.opname}({jump_labels[inst.argval]})\n"
                        )
                    elif inst.opname == "CALL":
                        f.write(f"; {inst.offset} {inst.opname}({inst.argval})\n")
                    else:
                        f.write(f"; {inst.offset} {inst.opname}({inst.argrepr})\n")

                if DEBUG:
                    for stack in types_stacks:
                        f.write(
                            "; TYPE_STACK: "
                            + " -> ".join([i.name for i in stack])
                            + "\n"
                        )

                match inst.opname:
                    case "STORE_NAME":
                        # TODO
                        pass
                    case "RESUME":
                        pass
                    case "LOAD_CONST":
                        match code_obj.co_consts[inst.arg]:
                            case None:
                                f.write("    lea rax, [None]\n")
                                types_stacks[-1].append(
                                    Type.from_builtin(BuiltinTypesEnum.NONE)
                                )
                                pass
                            case str():
                                f.write(
                                    f"    lea rax, [{function_base_name}_CONST_STRING{inst.arg}]\n"
                                )
                                types_stacks[-1].append(
                                    Type.from_builtin(BuiltinTypesEnum.STRING)
                                )
                            case int():
                                f.write(
                                    f"    lea rax, [{function_base_name}_CONST_INT{inst.arg}]\n"
                                )
                                types_stacks[-1].append(
                                    Type.from_builtin(BuiltinTypesEnum.INT)
                                )
                            case _:
                                raise Exception(
                                    f'Constant type "{type(inst.argval)}" is unimplemented'
                                )
                        f.write("    push rax\n")
                    case "STORE_FAST":
                        f.write("    pop rax\n")
                        f.write(f"    mov [rbp-8*{inst.arg + 1}], rax\n")
                        variables_types[inst.argval] = types_stacks[-1].pop()
                    case "LOAD_GLOBAL":
                        temp = more_itertools.seekable(dis.get_instructions(code_obj))
                        for i in temp:
                            if i.offset == inst.offset:
                                break
                        if code_obj.co_names[inst.arg // 2] in self.functions.keys():
                            function_arg_types = self._get_function_future_types(
                                code_obj, temp, variables_types
                            )
                        else:
                            function_arg_types = []
                        types_stacks[-1].append(
                            Type.from_builtin(
                                BuiltinTypesEnum.FUNCTION,
                                payload=builtin_functions.get(
                                    inst.argval,
                                    Type.from_builtin(BuiltinTypesEnum.UNKNOWN),
                                ),
                                function_name=code_obj.co_names[inst.arg // 2],
                                function_arg_types=function_arg_types,
                            )
                        )
                        _function_name = self._generate_function_name(
                            code_obj.co_names[inst.arg // 2], function_arg_types
                        )
                        f.write(f"    lea rax, {_function_name}\n")
                        f.write("    push rax\n")
                    case "LOAD_FAST":
                        f.write(f"    push qword [rbp-8*{inst.arg + 1}]\n")
                        types_stacks[-1].append(variables_types[inst.argval])
                    case "BINARY_OP":
                        f.write("    mov rax, [rsp+8]\n")
                        match inst.argrepr:
                            case "+" | "-" | "+=" | "-=" | "&":
                                f.write(
                                    f"    {bin_op_to_inst[inst.argrepr]} rax, [rsp]\n"
                                )
                            case "*":
                                f.write("    imul rax, [rsp]\n")
                            case "/":
                                f.write("    cqo\n")
                                f.write("    mov rcx, [rsp]\n")
                                f.write("    idiv rcx\n")
                            case _:
                                raise Exception(
                                    f"Instruction {inst.opname}({inst.argrepr}) is not implemented"
                                )
                        f.write("    add rsp, 16\n")
                        types_stacks[-1].pop()
                        types_stacks[-1].pop()
                        f.write("    push rax\n")
                        types_stacks[-1].append(Type.from_builtin(BuiltinTypesEnum.INT))
                    case "CALL":
                        for i in range(inst.arg):
                            f.write(
                                f"    pop {args_to_regs_map[inst.argval - i - 1]}\n"
                            )
                        f.write("    pop rax\n")
                        f.write("    mov rbx, rsp\n")
                        f.write("    and rsp, -16\n")
                        f.write("    call rax\n")
                        f.write("    mov rsp, rbx\n")
                        f.write("    push rax\n")
                        arg_types: list[Type] = []
                        for i in range(inst.arg):
                            arg_types.append(types_stacks[-1].pop())
                        function_type = types_stacks[-1].pop()
                        _function_name = self._generate_function_name(
                            function_type.function_name, arg_types
                        )
                        if (
                            _function_name not in self.codes_asm.keys()
                            and function_type.function_name
                            not in builtin_functions.keys()
                        ):
                            self.code_to_compile.append(
                                (self.functions[function_type.function_name], arg_types)
                            )
                        types_stacks[-1].append(function_type.payload)
                    case "POP_TOP":
                        f.write("    add rsp, 8\n")
                        types_stacks[-1].pop()
                    case "RETURN_CONST":
                        f.write(f"    add rsp, {(len(code_obj.co_varnames)) * 8}\n")
                        f.write("    pop rbx\n")
                        f.write("    pop rbp\n")
                        f.write(
                            f"    mov rax, {0 if inst.argval is None else inst.argval}\n"
                        )
                        f.write("    ret\n")
                    case "COMPARE_OP":
                        f.write("    mov rax, [rsp+8]\n")
                        f.write("    cmp rax, [rsp]\n")
                        f.write("    mov rax, 0\n")
                        f.write(f"    set{cmp_op_to_suffix[inst.argval]} al\n")
                        f.write("    add rsp, 16\n")
                        types_stacks[-1].pop()
                        types_stacks[-1].pop()
                        f.write("    push rax\n")
                        types_stacks[-1].append(Type.from_builtin(BuiltinTypesEnum.INT))
                    case "POP_JUMP_IF_FALSE":
                        f.write("    pop rax\n")
                        types_stacks[-1].pop()
                        f.write("    test rax, rax\n")
                        f.write(f"    jz {jump_labels[inst.argval]}\n")
                    case "JUMP_BACKWARD":
                        f.write(f"    jmp {jump_labels[inst.argval]}\n")
                    case "JUMP_FORWARD":
                        f.write(f"    jmp {jump_labels[inst.argval]}\n")
                    case "PUSH_NULL":
                        pass
                        # f.write("    push qword 0\n")
                    case "NOP":
                        pass
                    case _:
                        raise Exception(
                            f"Instruction {inst.opname}({inst.argrepr}) is not implemented"
                        )
            except Exception:
                raise Exception(
                    f"Compilation error during translating instruction {inst.offset} {inst.opname} ({inst.argrepr})"
                )
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
