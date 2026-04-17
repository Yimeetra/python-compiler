from __future__ import annotations
import dis
from types import CodeType


class Type:
    T_ANY: int = 0
    T_UNKNOWN: int = 1
    T_NONE: int = 2
    T_INT: int = 3

    def __init__(self, id) -> None:
        self.id: int = id

    def __eq__(self, other) -> bool:
        return self.id == other.id
    
    def __repr__(self) -> str:
        return f"type {self.id}"


TYPE_ANY = Type(Type.T_ANY)
TYPE_UNKNOWN = Type(Type.T_UNKNOWN)
TYPE_NONE = Type(Type.T_NONE)

class TypeInt(Type):
    def __init__(self, value: int | None = None) -> None:
        super().__init__(Type.T_INT)
        self.value: int | None = value
    
    def __repr__(self) -> str:
        return f"int {self.value}"


TYPE_INT = TypeInt()


class Node:
    ID: int = 0

    def __init__(
        self, inputs: list[Node] | None = None, type: Type = TYPE_UNKNOWN
    ) -> None:
        self.id: int = self.ID
        Node.ID += 1
        self.inputs: list[Node] = inputs or []
        self.outputs: list[Node] = []
        self.type = type

        for node in self.inputs:
            node.outputs.append(self)

    def __repr__(self) -> str:
        return f"Node {self.id}"
    
    def peephole(self) -> Node:
        return self

    def remove(self) -> None:
        for node in self.inputs:
            node.outputs.remove(self)

        for node in self.outputs:
            node.inputs.remove(self)

    def is_control(self) -> bool:
        raise Exception("Node base class kind is not known.")


class NodeStart(Node):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "START"

    def is_control(self) -> bool:
        return True


class NodeConstant(Node):
    def __init__(self, start_node: Node, type: Type) -> None:
        super().__init__([start_node])
        self.type = type

    def __repr__(self) -> str:
        return f"CONST {self.id}"

    def is_control(self) -> bool:
        return False


class NodeReturn(Node):
    def __init__(self, ctrl: Node, data: Node) -> None:
        super().__init__([ctrl, data])

    def __repr__(self) -> str:
        return "RETURN"

    def is_control(self) -> bool:
        return True


class NodeScope(Node):
    def __init__(self, start_node: Node) -> None:
        super().__init__([start_node])
        self.vars: dict[str, Node] = {}

    def __repr__(self) -> str:
        return f"SCOPE {self.vars}"


class NodeAdd(Node):
    def __init__(self, lhs: Node, rhs: Node):
        super().__init__([lhs, rhs])

    @property
    def lhs(self):
        return self.inputs[0]
    
    @property
    def rhs(self):
        return self.inputs[1]

    def __repr__(self):
        return "+"
    
    def peephole(self):
        if isinstance(self.lhs.type, TypeInt) and isinstance(self.rhs.type, TypeInt):
            if self.lhs.type.value is not None and self.rhs.type.value is not None:
                self.remove()
                return NodeConstant(lhs.inputs[0], self.lhs.type.value + self.rhs.type.value).peephole()


if __name__ == "__main__":

    def example():
        a = 1
        b = 2
        c = 0
        b = 3
        c = a + b
        return c

    code_obj = example.__code__
    nodes: dict[int, Node] = {}
    consts: dict[int, Node] = {}

    START_NODE = NodeStart()
    nodes[START_NODE.id] = START_NODE

    SCOPE_NODE = NodeScope(START_NODE)
    nodes[SCOPE_NODE.id] = SCOPE_NODE

    prev_ctrl_node = START_NODE

    for i, const in enumerate(code_obj.co_consts):
        match const:
            case int():
                const_node = NodeConstant(START_NODE, TypeInt(const)).peephole()
            case None:
                const_node = NodeConstant(START_NODE, TYPE_NONE).peephole()
            case _:
                raise Exception(
                    f"Constant type {const.__class__.__name__} is not implemented"
                )
        nodes[const_node.id] = const_node
        consts[i] = const_node

    stack: list[Node] = []

    for inst in dis.get_instructions(code_obj):
        match inst.opname:
            case "RESUME":
                pass
            case "RETURN_CONST":
                const_node = [
                    node
                    for node in START_NODE.outputs
                    if isinstance(node, NodeConstant) and node.type == inst.arg
                ][0]
                return_node = NodeReturn(START_NODE, const_node).peephole()
                nodes[const_node.id] = const_node
                nodes[return_node.id] = return_node
            case "LOAD_CONST":
                assert isinstance(inst.arg, int)

                stack.append(consts[inst.arg])
            case "STORE_FAST":
                data_node = stack.pop()
                SCOPE_NODE.vars[inst.argval] = data_node
            case "LOAD_FAST":
                stack.append(SCOPE_NODE.vars[inst.argval])
            case "RETURN_VALUE":
                data_node = stack.pop()
                return_node = NodeReturn(START_NODE, data_node).peephole()
                nodes[return_node.id] = return_node
            case "BINARY_OP":
                match inst.argrepr:
                    case "+":
                        lhs = stack.pop()
                        rhs = stack.pop()
                        add_node = NodeAdd(lhs, rhs).peephole()
                        nodes[add_node.id] = add_node
                        stack.append(add_node)
                    case _:
                        raise Exception(
                            f"Binary operation {inst.argrepr} is unimplemented."
                        )
            case _:
                raise Exception(f"{inst.opname} is unimplemented.")

    def print_ir(start_node: Node):
        nodes = find_all_nodes(start_node)
        scope_node = [node for node in nodes if isinstance(node, NodeScope)][0]

        print("digraph {")
        for node in nodes:
            if isinstance(node, NodeScope):
                continue
            print(f'{node.id} [ label = "{node}: {node.type}" ]')

        for node in nodes:
            for o_node in node.outputs:
                if isinstance(o_node, NodeScope):
                    continue
                print(f"{node.id} -> {o_node.id}")

        print(f"subgraph cluster_scope{scope_node.id} {{")
        print("node [shape=box]")
        for var, node in scope_node.vars.items():
            print(f'scope_{scope_node.id}_{var} [ label = "{var}" ]')
        print("}")
        for var, node in scope_node.vars.items():
            print(
                f"scope_{scope_node.id}_{var} -> {node.id} [style=dashed, color=blue]"
            )

        print("}")

    def find_all_nodes(start_node: Node) -> list[Node]:
        all: dict[int, Node] = {}
        walk(start_node, all)
        return list(all.values())

    def walk(node: Node, all: dict[int, Node]):
        if all.get(node.id) is not None:
            return
        all[node.id] = node
        for n in node.inputs:
            walk(n, all)
        for n in node.outputs:
            walk(n, all)

    print_ir(START_NODE)
