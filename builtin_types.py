from type import Type, BuiltinTypesEnum
from typing import Iterable, Callable
from types import CodeType
from dataclasses import dataclass, field
from ir import Frame


@dataclass
class Environment:
    code_obj: CodeType
    variable_types: dict[str, Type] = field(default_factory=dict)
    temp_type: Type = Type.from_builtin(BuiltinTypesEnum.unknown)
    last_arg_types: list[Type] = field(default_factory=list)


@dataclass(unsafe_hash=True)
class Function:
    base_name: str
    arg_types: tuple[Type, ...]
    _return_type_getter: Callable[[list[Type]], Type] = field(
        hash=False,
        compare=False,
        default=lambda x: Type.from_builtin(BuiltinTypesEnum.unknown),
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
        return self._return_type_getter(self.arg_types)


class BuiltInFunction(Function):
    def generate_function_name(self) -> str:
        return self.base_name


@dataclass(kw_only=True, unsafe_hash=True)
class CustomFunction(Function):
    env: Environment = field(hash=False)
    frames: list[Frame] = field(default_factory=list, hash=False)


class DunderWrapperFunction(Function):
    def generate_function_name(self) -> str:
        return f"{self.arg_types[0].name}__{self.base_name}__"

    def validate_args(self, input_arg_types):
        if type_has_method(input_arg_types[0], f"__{self.base_name}__"):
            self.arg_types = input_arg_types
        else:
            raise Exception(
                f"Type '{input_arg_types[0]}' doesn't implement method '__{self.base_name}__'"
            )

    def get_return_type(self):
        method = methods_of_type[self.arg_types[0]][f"__{self.base_name}__"]
        method.arg_types = self.arg_types
        return method.get_return_type()


class BuiltInMethod(Function):
    def generate_function_name(self) -> str:
        return (
            f"{self.arg_types[0].sub_types[0].name if self.arg_types[0].sub_types else ''}"
            + f"{'_' if self.arg_types[0].sub_types else ''}"
            + f"{self.arg_types[0].name}{self.base_name}"
        )


builtin_functions: dict[str, Function] = {
    "_print": (
        BuiltInFunction(
            "_print",
            (Type.from_builtin(BuiltinTypesEnum.str),),
            lambda _: Type.from_builtin(BuiltinTypesEnum.none),
        )
    ),
    "str": (
        DunderWrapperFunction(
            "str", (), lambda _: Type.from_builtin(BuiltinTypesEnum.str)
        )
    ),
    "len": (
        DunderWrapperFunction(
            "len", (), lambda _: Type.from_builtin(BuiltinTypesEnum.int)
        )
    ),
    "iter": (
        DunderWrapperFunction(
            "iter", (), lambda _: Type.from_builtin(BuiltinTypesEnum.iterator)
        )
    ),
    "next": (
        DunderWrapperFunction(
            "next", (), lambda _: Type.from_builtin(BuiltinTypesEnum.unknown)
        )
    ),
    "id": (
        BuiltInFunction(
            "id",
            (Type.from_builtin(BuiltinTypesEnum.any),),
            lambda _: Type.from_builtin(BuiltinTypesEnum.int),
        )
    ),
    "type": (
        BuiltInFunction(
            "type",
            (Type.from_builtin(BuiltinTypesEnum.any),),
            lambda _: Type.from_builtin(BuiltinTypesEnum.type),
        )
    ),
}


def binop_function(name: str, arg1: Type, arg2: Type, return_type: Type) -> Function:
    return BuiltInMethod(name, (arg1, arg2), lambda _: return_type)


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
        lambda _: Type.from_builtin(BuiltinTypesEnum.str),
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
        lambda _: Type.from_builtin(BuiltinTypesEnum.str),
    ),
}

list_methods: dict[str, Function] = {
    "__len__": BuiltInMethod(
        "__len__",
        (Type.from_builtin(BuiltinTypesEnum.list),),
        lambda _: Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__getitem__": BuiltInMethod(
        "__getitem__",
        (Type.from_builtin(BuiltinTypesEnum.list),),
        lambda self_types: self_types[0].sub_types[0],
    ),
    "__iter__": BuiltInMethod(
        "__iter__",
        (Type.from_builtin(BuiltinTypesEnum.list),),
        lambda self_types: Type.from_builtin(
            BuiltinTypesEnum.list_iterator,
            sub_types=(
                self_types[0].sub_types[0],
            ),  # TODO: temporarily return type is based on first element type
        ),
    ),
}

tuple_methods: dict[str, Function] = {
    "__len__": BuiltInMethod(
        "__len__",
        (Type.from_builtin(BuiltinTypesEnum.tuple),),
        lambda _: Type.from_builtin(BuiltinTypesEnum.int),
    ),
    "__getitem__": BuiltInMethod(
        "__getitem__",
        (Type.from_builtin(BuiltinTypesEnum.tuple),),
        lambda self_types: self_types[0].sub_types[0],
    ),
    "__iter__": BuiltInMethod(
        "__iter__",
        (Type.from_builtin(BuiltinTypesEnum.tuple),),
        lambda self_types: Type.from_builtin(
            BuiltinTypesEnum.tuple_iterator,
            sub_types=(
                self_types[0].sub_types[0],
            ),  # TODO: temporarily return type is based on first element type
        ),
    ),
}

list_iterator_methods: dict[str, Function] = {
    "__next__": BuiltInMethod(
        "__next__",
        (Type.from_builtin(BuiltinTypesEnum.list_iterator),),
        lambda self_types: self_types[0].sub_types[
            0
        ],  # TODO: temporarily return type is based on first element type
    ),
}

tuple_iterator_methods: dict[str, Function] = {
    "__next__": BuiltInMethod(
        "__next__",
        (Type.from_builtin(BuiltinTypesEnum.tuple_iterator),),
        lambda self_types: self_types[0].sub_types[
            0
        ],  # TODO: temporarily return type is based on first element type
    ),
}

type_methods: dict[str, Function] = {
    "__str__": BuiltInMethod(
        "__str__",
        (Type.from_builtin(BuiltinTypesEnum.type),),
        lambda _: Type.from_builtin(BuiltinTypesEnum.str),
    ),
    "__eq__": binop_function(
        "__eq__",
        Type.from_builtin(BuiltinTypesEnum.type),
        Type.from_builtin(BuiltinTypesEnum.type),
        Type.from_builtin(BuiltinTypesEnum.int),
    ),
}


methods_of_type: dict[Type, dict[str, Function]] = {
    Type.from_builtin(BuiltinTypesEnum.int): int_methods,
    Type.from_builtin(BuiltinTypesEnum.str): str_methods,
    Type.from_builtin(BuiltinTypesEnum.list): list_methods,
    Type.from_builtin(BuiltinTypesEnum.tuple): tuple_methods,
    Type.from_builtin(BuiltinTypesEnum.list_iterator): list_iterator_methods,
    Type.from_builtin(BuiltinTypesEnum.tuple_iterator): tuple_iterator_methods,
    Type.from_builtin(BuiltinTypesEnum.type): type_methods,
}

obj_name_to_type: dict[str, Type] = {
    "int": Type.from_builtin(BuiltinTypesEnum.int),
    "str": Type.from_builtin(BuiltinTypesEnum.str),
    "list": Type.from_builtin(BuiltinTypesEnum.list),
    "tuple": Type.from_builtin(BuiltinTypesEnum.tuple),
    "NoneType": Type.from_builtin(BuiltinTypesEnum.none),
}


def type_has_method(type: Type, method: str) -> bool:
    type_copy = type.copy()
    methods = methods_of_type.get(type_copy)
    if not methods:
        type_copy.sub_types = ()
        methods = methods_of_type.get(type_copy)
    if not methods:
        return False
    if not methods.get(method):
        return False
    return True
