from __future__ import annotations

import dis
import io
from types import CodeType

import more_itertools
import itertools

from ir import ThreeAddressCode, Operation, SourceType, Source
from type import Type, BuiltinTypesEnum

DEBUG = True


def type_has_method(type: Type, method: str) -> bool:
    methods = type_methods.get(type)
    if methods.get(method):
        return True
    return False


args_to_regs_map = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]


class Function:
    def __init__(
        self, base_name: str, arg_types: tuple[Type], return_type: Type
    ) -> None:
        self.base_name = base_name
        self.arg_types = arg_types
        self.return_type = return_type

    def generate_function_name(self) -> str:
        return "_".join([self.base_name] + [type.name for type in self.arg_types])

    def validate_args(self, input_arg_types: tuple[Type]) -> bool:
        for i, j in zip(self.arg_types, input_arg_types):
            if i.name != j.name:
                return False
        return True


class BuiltInFunction(Function):
    def generate_function_name(self) -> str:
        return self.base_name


class BuiltInMethodMapFunction(Function):
    def generate_function_name(self) -> str:
        return f"{self.arg_types[0].name}__{self.base_name}__"

    def validate_args(self, input_arg_types) -> bool:
        if type_has_method(input_arg_types[0], f"__{self.base_name}__"):
            self.arg_types = input_arg_types
            return True
        return False


class BuiltInMethod(Function):
    def generate_function_name(self) -> str:
        return self.base_name


builtin_functions: dict[str, Function] = {
    "_print": (
        BuiltInFunction(
            "_print",
            (Type.from_builtin(BuiltinTypesEnum.str),),
            Type.from_builtin(BuiltinTypesEnum.none),
        )
    ),
    "str": (
        BuiltInMethodMapFunction("str", (), (Type.from_builtin(BuiltinTypesEnum.str)))
    ),
}


def binop_function(name: str, arg1: Type, arg2: Type, return_type: Type) -> Type:
    return BuiltInMethod(name, (arg1, arg2), return_type)


int_methods: dict[str, Function] = {
    "__add__": binop_function(
        "__add__",
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__sub__": binop_function(
        "__sub__",
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__mul__": binop_function(
        "__mul__",
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__div__": binop_function(
        "__div__",
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__str__": BuiltInMethod(
        "__str__",
        (Type.from_builtin(BuiltinTypesEnum.int),),
        Type.from_builtin(BuiltinTypesEnum.str),
    ),
}

str_methods: dict[str, Function] = {
    "__add__": binop_function(
        "__add__",
        Type.from_builtin(BuiltinTypesEnum.str),
        Type.from_builtin(BuiltinTypesEnum.str),
        Type.from_builtin(BuiltinTypesEnum.str),
    ),
    "__str__": BuiltInMethod(
        "__str__",
        (Type.from_builtin(BuiltinTypesEnum.str),),
        Type.from_builtin(BuiltinTypesEnum.str),
    ),
}

type_methods: dict[Type, dict[str, Function]] = {
    Type.from_builtin(BuiltinTypesEnum.int): int_methods,
    Type.from_builtin(BuiltinTypesEnum.str): str_methods,
}

obj_name_to_type: dict[str, Type] = {
    "int": Type.from_builtin(BuiltinTypesEnum.int),
    "str": Type.from_builtin(BuiltinTypesEnum.str),
}

op_str_to_op_type: dict[str, Operation] = {
    "+": Operation.ADD,
    "-": Operation.SUB,
    "*": Operation.MUL,
    "/": Operation.DIV,
}

op_type_to_method: dict[Operation, str] = {
    Operation.ADD: "__add__",
    Operation.SUB: "__sub__",
    Operation.MUL: "__mul__",
    Operation.DIV: "__div__",
}


def get_type_of_source(
    source: Source, code_obj: CodeType, variable_types: dict[str, Type], temp_type: Type
) -> Type:
    match source.type:
        case SourceType.CONST:
            obj_name = code_obj.co_consts[source.value].__class__.__name__
            var_type = obj_name_to_type[obj_name]
        case SourceType.LOCAL:
            var_name = code_obj.co_varnames[source.value]
            var_type = variable_types[var_name]
        case SourceType.TEMP:
            var_type = temp_type
        case _:
            raise Exception(f"Cannot get type from SourceType {source.type.name}")
    return var_type


def emit_source_to_reg(source: Source, code_obj: CodeType, reg: str = "rax"):
    match source.type:
        case SourceType.CONST:
            return f"    lea {reg}, [{code_obj.co_name}_CONST{source.value}]\n"
        case SourceType.LOCAL:
            return f"    mov {reg}, [rbp-{(source.value + 1) * 8}]\n"
        case SourceType.TEMP:
            if reg != "rax":
                return f"    mov {reg}, rax\n"
            return ""
        case _:
            raise Exception(f"Cannot load from {source}")


def emit_reg_to_source(source: Source, code_obj: CodeType, reg: str = "rax"):
    match source.type:
        case SourceType.LOCAL:
            return f"    mov [rbp-{(source.value + 1) * 8}], {reg}\n"
        case SourceType.TEMP:
            if reg != "rax":
                return f"    mov rax, {reg}\n"
            return ""
        case _:
            raise Exception(f"Cannot load to {source}")


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
                output.append(
                    ThreeAddressCode(
                        Operation.LABEL.name, Source(SourceType.LABEL, label)
                    )
                )
            match inst.opname:
                case "LOAD_CONST":
                    stack.append(Source(SourceType.CONST, inst.arg))
                case "STORE_FAST":
                    output.append(
                        ThreeAddressCode(
                            Operation.ASSIGN,
                            stack.pop(),
                            None,
                            Source(SourceType.LOCAL, inst.arg),
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
                        op_str = inst.argrepr.replace("=", "")
                        op = op_str_to_op_type[op_str]
                        output.append(ThreeAddressCode(op, var1, var2, var1))
                        stack.append(Source(SourceType.LOCAL, inst.argval))
                        next(instructions)
                    else:
                        temp_var = f"t{next(var_iter)}"
                        op = op_str_to_op_type[inst.argrepr]
                        output.append(
                            ThreeAddressCode(
                                op, var1, var2, Source(SourceType.TEMP, temp_var)
                            )
                        )
                        stack.append(Source(SourceType.TEMP, temp_var))
                case "CALL":
                    temp_var = f"t{next(var_iter)}"
                    for i in range(inst.argval):
                        output.append(ThreeAddressCode(Operation.ARG, stack.pop()))
                    output.append(
                        ThreeAddressCode(
                            Operation.CALL,
                            stack.pop(),
                            None,
                            Source(SourceType.TEMP, temp_var),
                        )
                    )
                    stack.append(Source(SourceType.TEMP, temp_var))
                case "COMPARE_OP":
                    var2 = stack.pop()
                    var1 = stack.pop()
                    temp_var = f"t{next(var_iter)}"
                    op = op_str_to_op_type[inst.argrepr]
                    output.append(
                        ThreeAddressCode(
                            op, var1, var2, Source(SourceType.TEMP, temp_var)
                        )
                    )
                    stack.append(Source(SourceType.TEMP, temp_var))
                case "POP_JUMP_IF_FALSE":
                    var1 = stack.pop()
                    output.append(
                        ThreeAddressCode(
                            Operation.GOTO_IF_FALSE,
                            var1,
                            None,
                            Source(SourceType.LABEL, jump_labels[inst.argval]),
                        )
                    )
                case "POP_TOP":
                    stack.pop()
                case "JUMP_BACKWARD":
                    output.append(
                        ThreeAddressCode(
                            Operation.GOTO,
                            None,
                            None,
                            Source(SourceType.LABEL, jump_labels[inst.argval]),
                        )
                    )
                case "RETURN_CONST":
                    output.append(
                        ThreeAddressCode(
                            Operation.RETURN, Source(SourceType.CONST, inst.argval)
                        )
                    )
                case "RESUME":
                    pass
                case _:
                    raise Exception(f"Instruction {inst.opname} is unimplemented")
        return output

    def compile_file(self):
        f = self.output_file
        f.write("BITS 64\n")
        f.write("default rel\n")
        f.write("extern _print\n")
        for type, methods in type_methods.items():
            for method_name, method in methods.items():
                f.write(f"extern {type.name}{method.generate_function_name()}\n")

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
        self,
        code_obj: CodeType,
        ir: list[ThreeAddressCode],
        arg_types: list[Type] | None = None,
    ) -> str:
        variables_types: dict[str, Type] = {}
        temp_type = Type.from_builtin(BuiltinTypesEnum.unknown)

        if arg_types:
            for var, type in zip(
                code_obj.co_varnames,
                itertools.chain(
                    arg_types,
                    itertools.cycle([Type.from_builtin(BuiltinTypesEnum.unknown)]),
                ),
            ):
                variables_types[var] = type
        else:
            for var in code_obj.co_varnames:
                variables_types[var] = Type.from_builtin(BuiltinTypesEnum.unknown)

        curr_function = Function(
            code_obj.co_name,
            arg_types or (),
            Type.from_builtin(BuiltinTypesEnum.unknown),
        )

        function_name = curr_function.generate_function_name()
        base_function_name = curr_function.base_name

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
                case Operation.ASSIGN:
                    var_type = get_type_of_source(
                        inst.arg1, code_obj, variables_types, temp_type
                    )

                    f.write(emit_source_to_reg(inst.arg1, code_obj))
                    f.write(emit_reg_to_source(inst.dest, code_obj))

                    variables_types[code_obj.co_varnames[inst.dest.value]] = var_type
                case Operation.ARG:
                    args: list[Source] = []
                    args.append(inst.arg1)
                    while instructions.peek().op != Operation.CALL:
                        args.append(next(instructions).arg1)

                    args.reverse()

                    for arg, reg in zip(args, args_to_regs_map):
                        var_type = get_type_of_source(
                            arg, code_obj, variables_types, temp_type
                        )
                        f.write(emit_source_to_reg(arg, code_obj, reg))
                        last_arg_types.append(var_type)
                case Operation.CALL:
                    func = builtin_functions.get(
                        inst.arg1.value,
                        Function(
                            inst.arg1.value,
                            tuple(last_arg_types),
                            Type.from_builtin(BuiltinTypesEnum.unknown),
                        ),
                    )

                    if not func.validate_args(last_arg_types):
                        supplied_type_names = [type.name for type in last_arg_types]
                        required_type_names = [type.name for type in func.arg_types]
                        raise Exception(
                            f"Function {func.base_name} requires {', '.join(required_type_names)}, but was supplied with {', '.join(supplied_type_names)}"
                        )

                    call_function_name = func.generate_function_name()
                    f.write("    mov rbx, rsp\n")
                    f.write("    and rsp, -16\n")
                    f.write(f"    call {call_function_name}\n")
                    f.write("    mov rsp, rbx\n")
                    if (
                        inst.arg1.value not in builtin_functions.keys()
                        and call_function_name not in self.codes_asm.keys()
                    ):
                        self.code_to_compile.append(
                            (self.functions[inst.arg1.value], last_arg_types)
                        )

                    last_arg_types = []
                    if inst.dest.type == SourceType.LOCAL:
                        f.write(f"    mov [rbp-{(inst.dest.value + 1) * 8}], rax\n")
                        variables_types[code_obj.co_varnames[inst.dest.value]] = (
                            func.return_type
                        )
                    else:
                        temp_type = func.return_type
                case Operation.GOTO:
                    raise Exception("Unimplemented")
                case Operation.GOTO_IF_FALSE:
                    raise Exception("Unimplemented")
                case Operation.RETURN:
                    # TODO
                    f.write(f"    add rsp, {len(code_obj.co_varnames) * 8}\n")
                    f.write("    pop rbx\n")
                    f.write("    pop rbp\n")
                    f.write("    ret\n")
                case Operation.LABEL:
                    raise Exception("Unimplemented")
                case Operation.ADD | Operation.SUB | Operation.MUL | Operation.DIV:
                    method = op_type_to_method[inst.op]

                    arg1_type = get_type_of_source(
                        inst.arg1, code_obj, variables_types, temp_type
                    )
                    arg2_type = get_type_of_source(
                        inst.arg2, code_obj, variables_types, temp_type
                    )

                    if not type_has_method(arg1_type, method):
                        raise Exception(
                            f"Type {arg1_type.name} doesn't implement {method} method."
                        )

                    if not type_methods[arg1_type][method].validate_args(
                        (arg1_type, arg2_type)
                    ):
                        raise Exception(
                            f"Method {method} of {arg1_type.name} type doesn't support {arg2_type.name} argument type."
                        )

                    return_type = type_methods[arg1_type][method].return_type

                    f.write(
                        emit_source_to_reg(inst.arg1, code_obj, args_to_regs_map[0])
                    )
                    f.write(
                        emit_source_to_reg(inst.arg2, code_obj, args_to_regs_map[1])
                    )

                    f.write("    mov rbx, rsp\n")
                    f.write("    and rsp, -16\n")
                    f.write(f"    call {arg1_type.name}{method}\n")
                    f.write("    mov rsp, rbx\n")

                    if inst.dest.type == SourceType.LOCAL:
                        f.write(f"    mov [rbp-{(inst.dest.value + 1) * 8}], rax\n")
                        variables_types[code_obj.co_varnames[inst.dest.value]] = (
                            return_type
                        )
                    else:
                        temp_type = return_type
                case _:
                    raise Exception(f"Instruction {inst.op} is unimplemented")
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
