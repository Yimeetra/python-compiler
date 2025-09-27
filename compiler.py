from __future__ import annotations

import dis
import io
from types import CodeType
from typing import Iterable

import more_itertools
import itertools

from ir import ThreeAddressCode, Operation, SourceType, Source
from type import Type, BuiltinTypesEnum

from dataclasses import dataclass, field

DEBUG = True
EMIT_IR = True


def type_has_method(type: Type, method: str) -> bool:
    type_copy = Type(type.name)
    methods = type_methods.get(type_copy)
    if not methods:
        return False
    if not methods.get(method):
        return False
    return True


args_to_regs_map = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]


@dataclass(unsafe_hash=True)
class Function:
    base_name: str
    arg_types: tuple[Type, ...]
    _return_type: Type = field(
        hash=False, compare=False, default=Type.from_builtin(BuiltinTypesEnum.unknown)
    )

    def generate_function_name(self) -> str:
        return "_".join([self.base_name] + [type.name for type in self.arg_types])

    def validate_args(self, input_arg_types: Iterable[Type]):
        for i, j in zip(self.arg_types, input_arg_types):
            if i.name != j.name and not i.name == BuiltinTypesEnum.any.name:
                required_type_names = [type.name for type in self.arg_types]
                supplied_type_names = [type.name for type in input_arg_types]
                raise Exception(
                    f"Function {self.base_name} requires {', '.join(required_type_names)}, but was supplied with {', '.join(supplied_type_names)}"
                )

    def get_return_type(self):
        return self._return_type


class BuiltInFunction(Function):
    def generate_function_name(self) -> str:
        return self.base_name


class BuiltInMethodMapFunction(Function):
    def generate_function_name(self) -> str:
        return f"{self.arg_types[0].name}__{self.base_name}__"

    def validate_args(self, input_arg_types):
        if type_has_method(input_arg_types[0], f"__{self.base_name}__"):
            self.arg_types = input_arg_types
        else:
            raise Exception(
                f"Type '{input_arg_types[0]}' doesn't implement method '__{self.base_name}__'"
            )


class BuiltInMethod(Function):
    def generate_function_name(self) -> str:
        return self.base_name


class GetItemMethod(Function):
    def generate_function_name(self) -> str:
        return self.base_name

    def get_return_type(self):
        return self.arg_types[0].sub_types[0]


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
    "len": (
        BuiltInMethodMapFunction("len", (), (Type.from_builtin(BuiltinTypesEnum.int)))
    ),
    "id": (
        BuiltInFunction(
            "id",
            (Type.from_builtin(BuiltinTypesEnum.any),),
            Type.from_builtin(BuiltinTypesEnum.int),
        )
    ),
}


def binop_function(name: str, arg1: Type, arg2: Type, return_type: Type) -> Function:
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
    "__lt__": binop_function(
        "__lt__",
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__gt__": binop_function(
        "__gt__",
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__le__": binop_function(
        "__le__",
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__ge__": binop_function(
        "__ge__",
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__eq__": binop_function(
        "__eq__",
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__ne__": binop_function(
        "__ne__",
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

list_methods: dict[str, Function] = {
    "__len__": BuiltInMethod(
        "__len__",
        (Type.from_builtin(BuiltinTypesEnum.list),),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__getitem__": GetItemMethod(
        "__getitem__",
        (Type.from_builtin(BuiltinTypesEnum.list),),
        Type.from_builtin(BuiltinTypesEnum.unknown),
    ),
}

tuple_methods: dict[str, Function] = {
    "__len__": BuiltInMethod(
        "__len__",
        (Type.from_builtin(BuiltinTypesEnum.list),),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__getitem__": GetItemMethod(
        "__getitem__",
        (Type.from_builtin(BuiltinTypesEnum.list),),
        Type.from_builtin(BuiltinTypesEnum.unknown),
    ),
}

type_methods: dict[Type, dict[str, Function]] = {
    Type.from_builtin(BuiltinTypesEnum.int): int_methods,
    Type.from_builtin(BuiltinTypesEnum.str): str_methods,
    Type.from_builtin(BuiltinTypesEnum.list): list_methods,
    Type.from_builtin(BuiltinTypesEnum.tuple): tuple_methods,
}

obj_name_to_type: dict[str, Type] = {
    "int": Type.from_builtin(BuiltinTypesEnum.int),
    "str": Type.from_builtin(BuiltinTypesEnum.str),
    "list": Type.from_builtin(BuiltinTypesEnum.list),
    "tuple": Type.from_builtin(BuiltinTypesEnum.tuple),
    "NoneType": Type.from_builtin(BuiltinTypesEnum.none),
}

op_str_to_op_type: dict[str, Operation] = {
    "+": Operation.ADD,
    "-": Operation.SUB,
    "*": Operation.MUL,
    "/": Operation.DIV,
    "<": Operation.LT,
    ">": Operation.GT,
    "<=": Operation.LE,
    ">=": Operation.GE,
    "==": Operation.EQ,
    "!=": Operation.NE,
}

op_type_to_method: dict[Operation, str] = {
    Operation.ADD: "__add__",
    Operation.SUB: "__sub__",
    Operation.MUL: "__mul__",
    Operation.DIV: "__div__",
    Operation.LT: "__lt__",
    Operation.GT: "__gt__",
    Operation.LE: "__le__",
    Operation.GE: "__ge__",
    Operation.EQ: "__eq__",
    Operation.NE: "__ne__",
    Operation.GET_ITEM: "__getitem__",
}


@dataclass
class Environment:
    code_obj: CodeType
    variable_types: dict[str, Type] = field(default_factory=dict)
    temp_type: Type = Type.from_builtin(BuiltinTypesEnum.unknown)
    last_arg_types: list[Type] = field(default_factory=list)


def get_type_of_source(source: Source, env: Environment) -> Type:
    match source.type:
        case SourceType.CONST:
            assert isinstance(source.value, int)
            obj_name: str = env.code_obj.co_consts[source.value].__class__.__name__
            var_type = obj_name_to_type[obj_name]
            if var_type.name == Type.from_builtin(BuiltinTypesEnum.tuple).name:
                tuple_types = []
                for item in env.code_obj.co_consts[source.value]:
                    item_name: str = item.__class__.__name__
                    item_type = obj_name_to_type[item_name]
                    tuple_types.append(item_type)
                var_type.sub_types = tuple(tuple_types)
        case SourceType.LOCAL:
            assert isinstance(source.value, int)
            var_name = env.code_obj.co_varnames[source.value]
            var_type = env.variable_types[var_name]
        case SourceType.TEMP:
            var_type = env.temp_type
        case _:
            raise Exception(f"Cannot get type from SourceType {source.type.name}")
    return var_type


def emit_source_to_reg(source: Source, code_obj: CodeType, reg: str = "rax"):
    match source.type:
        case SourceType.CONST:
            assert isinstance(source.value, int)
            return f"    lea {reg}, [{code_obj.co_name}_CONST{source.value}]\n"
        case SourceType.LOCAL:
            assert isinstance(source.value, int)
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
            assert isinstance(source.value, int)
            return f"    mov [rbp-{(source.value + 1) * 8}], {reg}\n"
        case SourceType.TEMP:
            if reg != "rax":
                return f"    mov rax, {reg}\n"
            return ""
        case _:
            raise Exception(f"Cannot load to {source}")


def emit_const_obj(env_name: str, obj, const_n: int):
    f = io.StringIO()

    f.write(f"{env_name}_CONST{const_n}: \n")
    match obj:
        case str():
            f.write(".value_p: dq .value\n")
            f.write('.value: db "' + obj.replace("\n", '", 0xA, 0xD, "') + '", 0\n')
        case int():
            f.write(f"dq {obj}\n")
        case tuple():
            f.write(".values_p: dq .value0_p\n")
            f.write(f".length: dq {len(obj)}\n")
            for i, _ in enumerate(obj):
                f.write(f".value{i}_p: dq .value{i}_CONST{const_n}\n")
            for i, item in enumerate(obj):
                f.write(emit_const_obj(f".value{i}", item, const_n))
        case None:
            pass
        case _:
            raise Exception(
                f"Constants with type '{type(obj).__name__}' are not supported."
            )

    return f.getvalue()


class Compiler:
    def __init__(self, input_file_name: str, output_file_name: str) -> None:
        self.input_file_name: str = input_file_name
        self.output_file_name: str = output_file_name
        self._label_generator = self._get_next_label()
        self.output_file = open(output_file_name, "w+")
        self.local_code_objs: dict[str, CodeType] = {}
        self.function_asms: dict[Function, str] = {}
        self.function_irs: dict[Function, list[ThreeAddressCode]] = {}
        self.function_envs: dict[Function, Environment] = {}
        self.fn_name_types_to_fn: dict[tuple[str, tuple[Type, ...]], Function] = {}

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
                    f.write(emit_const_obj(code.co_name, const, i))
        f.seek(0)
        return f.read()

    def compile_ir(self, code_obj) -> list[ThreeAddressCode]:
        code_obj

        instructions = more_itertools.seekable(dis.get_instructions(code_obj))
        var_iter = itertools.count()
        label_iter = itertools.count()

        jump_labels: dict[int, str] = {}

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
                    ThreeAddressCode(Operation.LABEL, Source(SourceType.LABEL, label))
                )
            match inst.opname:
                case "LOAD_CONST":
                    assert isinstance(inst.arg, int)
                    stack.append(Source(SourceType.CONST, inst.arg))
                case "STORE_FAST":
                    assert isinstance(inst.arg, int)
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
                    assert isinstance(inst.arg, int)
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
                    assert isinstance(inst.arg, int)
                    output.append(
                        ThreeAddressCode(
                            Operation.RETURN, Source(SourceType.CONST, inst.arg)
                        )
                    )
                case "RETURN_VALUE":
                    output.append(ThreeAddressCode(Operation.RETURN, stack.pop()))
                case "RESUME":
                    pass
                case "NOP":
                    pass
                case "BUILD_LIST":
                    assert isinstance(inst.arg, int)
                    temp_var = f"t{next(var_iter)}"

                    for i in range(inst.arg):
                        output.append(
                            ThreeAddressCode(Operation.VA_ARG, arg1=stack.pop())
                        )
                    output.append(
                        ThreeAddressCode(
                            Operation.BUILD_LIST, dest=Source(SourceType.TEMP, temp_var)
                        )
                    )
                    stack.append(Source(SourceType.TEMP, temp_var))
                case "BINARY_SUBSCR":
                    temp_var = f"t{next(var_iter)}"
                    index = stack.pop()
                    src = stack.pop()
                    output.append(
                        ThreeAddressCode(
                            Operation.GET_ITEM,
                            dest=Source(SourceType.TEMP, temp_var),
                            arg1=src,
                            arg2=index,
                        )
                    )
                    stack.append(Source(SourceType.TEMP, temp_var))
                case _:
                    raise Exception(f"Instruction {inst.opname} is unimplemented")
        return output

    def annotate_ir_types(
        self,
        ir: list[ThreeAddressCode],
        env: Environment,
        arg_types: list[Type] | None,
        start_at: int = 0,
    ) -> tuple[Type, list[ThreeAddressCode]]:
        arg_types = arg_types or []
        return_type = Type.from_builtin(BuiltinTypesEnum.unknown)

        for var, type in zip(
            env.code_obj.co_varnames[: env.code_obj.co_argcount],
            itertools.chain(
                arg_types,
                itertools.cycle([Type.from_builtin(BuiltinTypesEnum.unknown)]),
            ),
        ):
            env.variable_types[var] = type

        variadic_args_count = 0
        variadic_args_types: set[Type] = set()

        instructions = more_itertools.seekable(enumerate(ir))
        instructions.seek(start_at)

        for i, inst in instructions:
            match inst.op:
                case Operation.ASSIGN:
                    assert inst.arg1 is not None
                    assert inst.dest is not None
                    assert isinstance(inst.dest.value, int)

                    inst.dest_type = get_type_of_source(inst.arg1, env)
                    env.variable_types[env.code_obj.co_varnames[inst.dest.value]] = (
                        inst.dest_type
                    )
                case Operation.ARG:
                    assert inst.arg1 is not None

                    arg_type = get_type_of_source(inst.arg1, env)
                    env.last_arg_types.append(arg_type)
                    inst.dest_type = arg_type
                case Operation.CALL:
                    assert inst.arg1 is not None
                    assert inst.dest is not None
                    assert isinstance(inst.arg1.value, str)

                    func = builtin_functions.get(inst.arg1.value)
                    if func is None:
                        func = self.fn_name_types_to_fn.get(
                            (inst.arg1.value, tuple(env.last_arg_types))
                        )

                    if func is None:
                        self.compile_queue.append(
                            (i, Function(env.code_obj.co_name, tuple(arg_types)))
                        )
                        self.compile_queue.append(
                            (0, Function(inst.arg1.value, tuple(env.last_arg_types)))
                        )
                        return return_type, ir

                    func.validate_args(env.last_arg_types)

                    env.last_arg_types = []
                    if inst.dest.type == SourceType.LOCAL:
                        assert isinstance(inst.dest.value, int)
                        env.variable_types[
                            env.code_obj.co_varnames[inst.dest.value]
                        ] = func.get_return_type()
                    else:
                        env.temp_type = func.get_return_type()

                    inst.arg1.value = func.generate_function_name()
                    inst.dest_type = func.get_return_type()
                case Operation.GOTO:
                    pass
                case Operation.GOTO_IF_FALSE:
                    assert inst.arg1 is not None
                    inst.dest_type = get_type_of_source(inst.arg1, env)
                case Operation.RETURN:
                    assert inst.arg1 is not None
                    inst.dest_type = get_type_of_source(inst.arg1, env)
                    return_type = inst.dest_type
                case Operation.LABEL:
                    pass
                case (
                    Operation.ADD
                    | Operation.SUB
                    | Operation.MUL
                    | Operation.DIV
                    | Operation.LT
                    | Operation.GT
                    | Operation.LE
                    | Operation.GE
                    | Operation.EQ
                    | Operation.NE
                ):
                    assert inst.arg1 is not None
                    assert inst.arg2 is not None
                    assert inst.dest is not None

                    method = op_type_to_method[inst.op]

                    arg1_type = get_type_of_source(inst.arg1, env)
                    arg2_type = get_type_of_source(inst.arg2, env)

                    if not type_has_method(arg1_type, method):
                        raise Exception(
                            f"Type {arg1_type.name} doesn't implement {method} method."
                        )

                    type_methods[arg1_type][method].validate_args(
                        (arg1_type, arg2_type)
                    )

                    return_type = type_methods[arg1_type][method].get_return_type()

                    if inst.dest.type == SourceType.LOCAL:
                        assert isinstance(inst.dest.value, int)
                        env.variable_types[
                            env.code_obj.co_varnames[inst.dest.value]
                        ] = return_type
                    else:
                        env.temp_type = return_type
                    inst.dest_type = return_type
                    inst.arg1_type = arg1_type
                    inst.arg2_type = arg2_type
                case Operation.VA_ARG:
                    assert inst.arg1 is not None
                    variadic_args_count += 1
                    va_arg_type = get_type_of_source(inst.arg1, env)
                    variadic_args_types.add(va_arg_type)
                    inst.dest_type = va_arg_type
                case Operation.BUILD_LIST:
                    if len(variadic_args_types) > 1:
                        raise Exception("Allowed lists only of one type.")

                    if len(variadic_args_types) != 0:
                        list_type = variadic_args_types.pop()
                    else:
                        list_type = Type.from_builtin(BuiltinTypesEnum.none)

                    env.temp_type = Type.from_builtin(
                        BuiltinTypesEnum.list, (list_type,)
                    )
                    inst.dest_type = Type.from_builtin(
                        BuiltinTypesEnum.list, (list_type,)
                    )
                case Operation.GET_ITEM:
                    assert inst.arg1 is not None
                    container_type = get_type_of_source(inst.arg1, env)
                    index_type = get_type_of_source(inst.arg2, env)

                    if (
                        container_type.name
                        == Type.from_builtin(BuiltinTypesEnum.tuple).name
                    ):
                        if inst.arg2.type != SourceType.CONST:
                            raise Exception("Tuple index must be constant.")

                        if index_type != Type.from_builtin(BuiltinTypesEnum.int):
                            raise Exception("Tuple index must be 'int' type.")

                        index = env.code_obj.co_consts[inst.arg2.value]
                        inst.dest_type = container_type.sub_types[index]
                        inst.arg1_type = container_type
                        env.temp_type = container_type.sub_types[index]
                    else:
                        assert len(container_type.sub_types) > 0
                        inst.dest_type = container_type.sub_types[0]
                        inst.arg1_type = container_type
                        env.temp_type = container_type.sub_types[0]
                case _:
                    raise Exception(f"Instruction {inst.op} is unimplemented")

        return return_type, ir

    def compile_file(self):
        f = self.output_file
        f.write("BITS 64\n")
        f.write("default rel\n")
        f.write("extern _print\n")
        f.write("extern id\n")
        f.write("extern build_list\n")

        for type, methods in type_methods.items():
            for method_name, method in methods.items():
                f.write(f"extern {type.name}{method.generate_function_name()}\n")

        f.write("section .text\n")

        with open(self.input_file_name, "r", encoding="utf-8") as _f:
            code_obj = compile(_f.read(), self.input_file_name, "exec")

        main_code: CodeType | None

        function_name_to_env: dict[str, Environment] = {}
        function_name_to_ir: dict[str, list[ThreeAddressCode]] = {}

        for code in code_obj.co_consts:
            if not isinstance(code, CodeType):
                continue
            self.local_code_objs[code.co_name] = code

            ir = self.compile_ir(code)

            function_name_to_ir[code.co_name] = ir

            if code.co_name == "main":
                main_code = code

            if EMIT_IR:
                filename = code.co_name
                with open(f"{filename}.ir", "w") as ir_f:
                    for i in ir:
                        ir_f.write(f"{repr(i)}\n")

        if main_code is None:
            raise Exception("Function 'main' is undefined")

        self.compile_queue: list[tuple[int, Function]] = []

        self.compile_queue.append(
            (0, Function("main", (), Type.from_builtin(BuiltinTypesEnum.none)))
        )

        while len(self.compile_queue) > 0:
            start_at, func = self.compile_queue.pop()

            _env = self.function_envs.get(func)
            if not _env:
                code = self.local_code_objs[func.base_name]
                env = Environment(code)
                self.function_envs[func] = env
            else:
                env = _env

            _ir = self.function_irs.get(func)
            if not _ir:
                ir = function_name_to_ir[func.base_name].copy()
                for i, inst in enumerate(ir):
                    ir[i] = inst.copy()
            else:
                ir = _ir

            return_type, ir = self.annotate_ir_types(
                ir, env, list(func.arg_types), start_at
            )

            func._return_type = return_type

            self.fn_name_types_to_fn[(func.base_name, func.arg_types)] = func
            self.function_irs[func] = ir
            self.function_envs[func] = env

            if EMIT_IR:
                filename = func.generate_function_name()
                with open(f"{filename}.ir", "w") as ir_f:
                    for i in ir:
                        ir_f.write(f"{repr(i)}\n")

        for func, ir in self.function_irs.items():
            env = self.function_envs[func]
            self.compile_nasm(ir, env)

        for _, asm in self.function_asms.items():
            f.write(asm)

        f.write("section .data\n")

        f.write("global None\n")
        f.write("None: dq 0\n")
        f.write(self.generate_constants(code_obj))

    def compile_nasm(self, ir: list[ThreeAddressCode], env: Environment) -> str:
        arg_types = tuple(env.variable_types.values())[: env.code_obj.co_argcount]

        curr_function = Function(
            env.code_obj.co_name,
            arg_types,
            Type.from_builtin(BuiltinTypesEnum.unknown),
        )

        function_name = curr_function.generate_function_name()
        base_function_name = curr_function.base_name

        f = io.StringIO()

        f.write(f"global {function_name}\n")
        if DEBUG:
            temp: dict = {
                name: type.name
                for name, type in list(env.variable_types.items())[
                    : env.code_obj.co_argcount
                ]
            }
            f.write(f"{function_name}: ; {temp}\n")
        else:
            f.write(f"{function_name}:\n")

        f.write("    push rbp\n")
        f.write("    push rbx\n")
        f.write("    mov rbp, rsp\n")
        f.write(f"    sub rsp, {len(env.code_obj.co_varnames) * 8}\n")

        for i in range(env.code_obj.co_argcount):
            f.write(f"    mov [rbp-8*{i + 1}], {args_to_regs_map[i]}\n")

        instructions = more_itertools.seekable(enumerate(ir))

        last_arg_types: list[Type] = []
        variadic_args_count: int = 0

        args: list[Source]

        for i, inst in instructions:
            if inst.dest_type == Type.from_builtin(
                BuiltinTypesEnum.unknown
            ) and inst.op not in [
                Operation.GOTO_IF_FALSE,
                Operation.GOTO,
                Operation.LABEL,
                Operation.CALL,  # TODO: CALL is temporary
            ]:
                raise Exception(f"Instruction's {i} {inst.op.name} type is unknown")
            if DEBUG:
                f.write(f"; {inst}\n")
            match inst.op:
                case Operation.ASSIGN:
                    assert inst.arg1 is not None
                    assert inst.dest is not None
                    assert isinstance(inst.dest.value, int)

                    f.write(emit_source_to_reg(inst.arg1, env.code_obj))
                    f.write(emit_reg_to_source(inst.dest, env.code_obj))
                case Operation.ARG:
                    assert inst.arg1 is not None

                    args = []
                    args.append(inst.arg1)
                    while instructions.peek()[1].op == Operation.ARG:
                        arg = next(instructions)[1].arg1
                        assert arg is not None
                        args.append(arg)

                    args.reverse()

                    for arg, reg in zip(args, args_to_regs_map):
                        f.write(emit_source_to_reg(arg, env.code_obj, reg))
                case Operation.CALL:
                    assert inst.dest is not None
                    assert inst.arg1 is not None
                    assert isinstance(inst.arg1.value, str)

                    func = builtin_functions.get(
                        inst.arg1.value,
                        Function(
                            inst.arg1.value,
                            tuple(last_arg_types),
                            Type.from_builtin(BuiltinTypesEnum.unknown),
                        ),
                    )

                    call_function_name = func.generate_function_name()
                    f.write("    mov rbx, rsp\n")
                    f.write("    and rsp, -16\n")
                    f.write(f"    call {call_function_name}\n")
                    f.write("    mov rsp, rbx\n")

                    if inst.dest.type == SourceType.LOCAL:
                        assert isinstance(inst.dest.value, int)
                        f.write(f"    mov [rbp-{(inst.dest.value + 1) * 8}], rax\n")
                case Operation.GOTO:
                    assert inst.dest is not None
                    f.write(f"    jmp .{inst.dest.value}\n")
                case Operation.GOTO_IF_FALSE:
                    assert inst.arg1 is not None
                    assert inst.dest is not None
                    f.write(emit_source_to_reg(inst.arg1, env.code_obj))
                    f.write("    mov rax, [rax]\n")
                    f.write("    test rax, rax\n")
                    f.write(f"    jz .{inst.dest.value}\n")
                case Operation.RETURN:
                    assert inst.arg1 is not None

                    f.write(emit_source_to_reg(inst.arg1, env.code_obj))
                    f.write(f"    add rsp, {len(env.code_obj.co_varnames) * 8}\n")
                    f.write("    pop rbx\n")
                    f.write("    pop rbp\n")
                    f.write("    ret\n")
                case Operation.LABEL:
                    assert inst.arg1 is not None
                    f.write(f".{inst.arg1.value}:\n")
                case (
                    Operation.ADD
                    | Operation.SUB
                    | Operation.MUL
                    | Operation.DIV
                    | Operation.LT
                    | Operation.GT
                    | Operation.LE
                    | Operation.GE
                    | Operation.EQ
                    | Operation.NE
                    | Operation.GET_ITEM
                ):
                    assert inst.arg1 is not None
                    assert inst.arg2 is not None
                    assert inst.dest is not None

                    method = op_type_to_method[inst.op]

                    f.write(
                        emit_source_to_reg(inst.arg1, env.code_obj, args_to_regs_map[0])
                    )
                    f.write(
                        emit_source_to_reg(inst.arg2, env.code_obj, args_to_regs_map[1])
                    )
                    f.write("    mov rbx, rsp\n")
                    f.write("    and rsp, -16\n")
                    f.write(f"    call {inst.arg1_type.name}{method}\n")
                    f.write("    mov rsp, rbx\n")

                    if inst.dest.type == SourceType.LOCAL:
                        assert isinstance(inst.dest.value, int)
                        f.write(f"    mov [rbp-{(inst.dest.value + 1) * 8}], rax\n")
                case Operation.VA_ARG:
                    assert inst.arg1 is not None

                    args = []
                    args.append(inst.arg1)
                    variadic_args_count += 1
                    while instructions.peek()[1].op == Operation.VA_ARG:
                        variadic_args_count += 1
                        arg = next(instructions)[1].arg1
                        assert arg is not None
                        args.append(arg)

                    args.reverse()

                    for arg, reg in zip(args, args_to_regs_map[1:]):
                        f.write(emit_source_to_reg(arg, env.code_obj, reg))

                    assert inst.arg1 is not None
                case Operation.BUILD_LIST:
                    f.write(f"    mov rdi, {variadic_args_count}\n")
                    f.write("    mov rbx, rsp\n")
                    f.write("    and rsp, -16\n")
                    f.write("    call build_list\n")
                    f.write("    mov rsp, rbx\n")
                    variadic_args_count = 0
                case _:
                    raise Exception(f"Instruction {inst.op} is unimplemented")
        f.seek(0)
        self.function_asms[curr_function] = f.read()
        return function_name


if __name__ == "__main__":
    compiler = Compiler("main.py", "main.asm")

    with open(compiler.input_file_name, "r", encoding="utf-8") as _f:
        code_obj = compile(_f.read(), compiler.input_file_name, "exec")

    dis.dis(code_obj)

    compiler.compile_file()
