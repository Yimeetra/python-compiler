import dis
from types import CodeType
import more_itertools
import itertools
import io

cmp_op_to_suffix = {
    ">": "g",
    "<": "l",
    ">=": "ge",
    "<=": "le",
    "==": "e",
    "!=": "ne"
}

bin_op_to_inst = {
    "+": "add",
    "-": "sub", 
    "+=": "add",
    "-=": "sub",
    "&": "and"
}

args_to_regs_map = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]

class Compiler:
    def __init__(self, input_file_name: str, output_file_name: str) -> None:
        self.input_file_name: str = input_file_name
        self.output_file_name: str = output_file_name
        self.jump_labels: dict[int, str] = {}
        self._label_generator = self._get_next_label()
        self.output_file = open(output_file_name, "w+")
        self.codes_asm: dict[str, str] = {}
    
    def __del__(self):
        self.output_file.close()

    def _get_next_label(self):
        i = 0
        while 1:
            yield f".L{i}"
            i += 1

    def compile_file(self):
        f = self.output_file
        f.write("BITS 64\n")
        f.write("default rel\n")
        f.write("extern print_int\n")
        f.write("extern print_string\n")
        f.write("extern print\n")

        f.write("section .text\n")
        # f.write("global _start\n")
        # f.write("_start:\n")

        with open(self.input_file_name, "r", encoding="utf-8") as _f:
            code_obj = compile(_f.read(), self.input_file_name, "exec")

        for code in code_obj.co_consts:
            if not isinstance(code, CodeType):
                continue
            print("--------------------------")
            dis.show_code(code)
            self.compile_code(code)
            f.write(self.codes_asm[code.co_name])
        
        f.write("section .data\n")

        for code in code_obj.co_consts:
            if isinstance(code, CodeType):
                for i, const in enumerate(code.co_consts):
                    if isinstance(const, str):
                        f.write(f"{code.co_name}_CONST{i}: ")
                        f.write(f"db \"{const.replace("\n", "\", 0xA, 0xD, \"")}\", 0\n")

    def compile_code(self, code_obj: CodeType) -> None:
        f = io.StringIO()
        jump_labels = {}
        instructions = more_itertools.seekable(dis.get_instructions(code_obj))
        for inst in instructions:
            if "JUMP" in inst.opname:
                jump_labels[inst.argval] = next(self._label_generator)

        f.write(f"global {code_obj.co_name}\n")
        f.write(f"{code_obj.co_name}:\n")

        f.write("    push rbp\n")
        f.write("    mov rbp, rsp\n")
        f.write(f"    sub rsp, {len(code_obj.co_varnames)*8}\n")

        for i in range(code_obj.co_argcount):
            f.write(f"    mov [rbp-8*{i+1}], {args_to_regs_map[i]}\n")
            

        instructions.seek(0)
        for inst in instructions:
            label = jump_labels.get(inst.offset)
            if label:
                f.write(f"{label}:\n")
            match inst.opname:
                case "STORE_NAME":
                    # TODO
                    pass
                case "RESUME":
                    pass
                case "LOAD_CONST":
                    f.write(f"; {inst.offset} LOAD_CONST\n")
                    match code_obj.co_consts[inst.arg]:
                        case None:
                            f.write("    push qword 0\n")
                        case str():
                            f.write(f"    lea rax, {code_obj.co_name}_CONST{inst.arg}\n")
                            f.write("    push rax\n")
                        case CodeType():
                            if next(instructions).opname == "MAKE_FUNCTION" and instructions.peek().opname != "STORE_NAME":
                                f.write(f"    lea rax, {code_obj.co_names[inst.arg-1]}\n")
                        case int():
                            f.write(f"    push qword {inst.argval}\n")
                        case _:
                            raise Exception(f"Constant type \"{type(inst.argval)}\" is unimplemented")
                case "STORE_FAST":
                    f.write(f"; {inst.offset} STORE_FAST\n")
                    f.write("    pop rax\n")
                    f.write(f"    mov [rbp-8*{inst.arg+1}], rax\n")
                case "LOAD_GLOBAL":
                    f.write(f"; {inst.offset} LOAD_GLOBAL\n")
                    f.write(f"    lea rax, {code_obj.co_names[inst.arg-1]}\n")
                    f.write("    push rax\n")
                case "LOAD_FAST":
                    f.write(f"; {inst.offset} LOAD_FAST\n")
                    f.write(f"    push qword [rbp-8*{inst.arg+1}]\n")
                case "BINARY_OP":
                    f.write(f"; {inst.offset} BINARY_OP\n")
                    f.write("    mov rax, [rsp+8]\n")
                    match inst.argrepr:
                        case "+"|"-"|"+="|"-="|"&":
                            f.write(f"    {bin_op_to_inst[inst.argrepr]} rax, [rsp]\n")
                        case "*":
                            f.write("    imul rax, [rsp]\n")
                        case "/":
                            f.write("    cqo\n")
                            f.write("    mov rcx, [rsp]\n")
                            f.write("    idiv rcx\n")
                        case _:
                            raise Exception(f"Instruction {inst.opname}({inst.argrepr}) is not implemented")
                    f.write("    add rsp, 16\n")
                    f.write("    push rax\n")
                case "CALL":
                    f.write(f"; {inst.offset} CALL\n")
                    for i in range(inst.arg):
                        f.write(f"    pop {args_to_regs_map[i]}\n")
                    f.write("    pop rax\n")
                    f.write("    call rax\n")
                    f.write("    push rax\n")
                case "POP_TOP":
                    f.write(f"; {inst.offset} POP_TOP\n")
                    f.write("    add rsp, 8\n")
                case "RETURN_CONST":
                    f.write(f"; {inst.offset} RETURN_CONST\n")
                    f.write(f"    add rsp, {(len(code_obj.co_varnames))*8}\n")
                    f.write("    pop rbp\n")
                    f.write(f"    mov rax, {0 if inst.argval is None else inst.argval}\n")
                    f.write("    ret\n")
                case "COMPARE_OP":
                    f.write(f"; {inst.offset} COMPARE_OP\n")
                    f.write("    mov rax, [rsp+8]\n")
                    f.write("    cmp rax, [rsp]\n")
                    f.write("    mov rax, 0\n")
                    f.write(f"    set{cmp_op_to_suffix[inst.argval]} al\n")
                    f.write("    add rsp, 16\n")
                    f.write("    push rax\n")
                case "POP_JUMP_IF_FALSE":
                    f.write(f"; {inst.offset} POP_JUMP_IF_FALSE({jump_labels[inst.argval]})\n")
                    f.write("    pop rax\n")
                    f.write("    test rax, rax\n")
                    f.write(f"    jz {jump_labels[inst.argval]}\n")
                case "JUMP_BACKWARD":
                    f.write(f"; {inst.offset} JUMP_BACKWARD({jump_labels[inst.argval]})\n")
                    f.write(f"    jmp {jump_labels[inst.argval]}\n")
                case "JUMP_FORWARD":
                    f.write(f"; {inst.offset} JUMP_FORWARD({jump_labels[inst.argval]})\n")
                    f.write(f"    jmp {jump_labels[inst.argval]}\n")
                case _:
                    raise Exception(f"Instruction {inst.opname}({inst.argrepr}) is not implemented")
        f.seek(0)
        self.codes_asm[code_obj.co_name] = f.read()

if __name__ == "__main__":
    with open("main.py", "r", encoding="utf-8") as f:
        code_obj = compile(f.read(), "main.py", "exec")
    dis.dis(code_obj)
    dis.show_code(code_obj)

    compiler = Compiler("main.py", "main.asm")
    compiler.compile_file()

"""

f.write(f"; {inst.offset} MAKE_FUNCTION\n")
f.write(f"global {inst.argrepr}\n")
f.write(f"{inst.argrepr}:\n")

f.write("    push rbp\n")
f.write("    mov rbp, rsp\n")
f.write(f"    sub rsp, {len(code_obj.co_varnames)*8}\n")

"""