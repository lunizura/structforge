from __future__ import annotations

from structforge.models import (
    FieldKind,
    PointerChain,
    SdkSpec,
    StructDef,
    StructField,
    validate_spec,
)


def _py_type(kind: FieldKind) -> str:
    return {
        FieldKind.VPTR: "int",
        FieldKind.PTR64: "int",
        FieldKind.CSTR_PTR: "int",
        FieldKind.STD_VECTOR: "RawVectorView",
        FieldKind.F64: "float",
        FieldKind.I64: "int",
        FieldKind.U64: "int",
        FieldKind.F32: "float",
        FieldKind.I32: "int",
        FieldKind.U32: "int",
        FieldKind.U16: "int",
        FieldKind.U8: "int",
        FieldKind.BOOL: "bool",
        FieldKind.CHAR_ARRAY: "str",
        FieldKind.PAD: "bytes",
    }[kind]


def _unpack_expr(fld: StructField) -> str:
    off = f"offset + 0x{fld.offset:x}"
    if fld.kind in (FieldKind.VPTR, FieldKind.PTR64, FieldKind.CSTR_PTR, FieldKind.U64):
        return f"struct.unpack_from('<Q', buf, {off})[0]"
    if fld.kind == FieldKind.I64:
        return f"struct.unpack_from('<q', buf, {off})[0]"
    if fld.kind == FieldKind.F64:
        return f"struct.unpack_from('<d', buf, {off})[0]"
    if fld.kind == FieldKind.F32:
        return f"struct.unpack_from('<f', buf, {off})[0]"
    if fld.kind == FieldKind.I32:
        return f"struct.unpack_from('<i', buf, {off})[0]"
    if fld.kind == FieldKind.U32:
        return f"struct.unpack_from('<I', buf, {off})[0]"
    if fld.kind == FieldKind.U16:
        return f"struct.unpack_from('<H', buf, {off})[0]"
    if fld.kind == FieldKind.U8:
        return f"buf[{off}]"
    if fld.kind == FieldKind.BOOL:
        return f"bool(buf[{off}])"
    if fld.kind == FieldKind.STD_VECTOR:
        return f"RawVectorView(*struct.unpack_from('<QQQ', buf, {off}))"
    if fld.kind == FieldKind.CHAR_ARRAY:
        return f"buf[{off} : {off} + {fld.size}].split(b'\\x00', 1)[0].decode('ascii', errors='replace')"
    return f"bytes(buf[{off} : {off} + {fld.size}])"


def _chain_unpack(kind: FieldKind) -> tuple[int, str, str]:
    return {
        FieldKind.VPTR: (8, "int", "struct.unpack('<Q', raw)[0]"),
        FieldKind.PTR64: (8, "int", "struct.unpack('<Q', raw)[0]"),
        FieldKind.CSTR_PTR: (8, "str", "_read_cstr_mem(read_mem, struct.unpack('<Q', raw)[0])"),
        FieldKind.STD_VECTOR: (24, "RawVectorView", "RawVectorView(*struct.unpack('<QQQ', raw))"),
        FieldKind.U64: (8, "int", "struct.unpack('<Q', raw)[0]"),
        FieldKind.I64: (8, "int", "struct.unpack('<q', raw)[0]"),
        FieldKind.F64: (8, "float", "struct.unpack('<d', raw)[0]"),
        FieldKind.F32: (4, "float", "struct.unpack('<f', raw)[0]"),
        FieldKind.I32: (4, "int", "struct.unpack('<i', raw)[0]"),
        FieldKind.U32: (4, "int", "struct.unpack('<I', raw)[0]"),
        FieldKind.U16: (2, "int", "struct.unpack('<H', raw)[0]"),
        FieldKind.U8: (1, "int", "raw[0]"),
        FieldKind.BOOL: (1, "bool", "bool(raw[0])"),
    }[kind]


def _emit_py_struct(st: StructDef) -> list[str]:
    lines: list[str] = [
        "@dataclass",
        f"class {st.name}:",
    ]
    for fld in st.fields:
        lines.append(f"    {fld.name}: {_py_type(fld.kind)}")
    lines.append("")
    lines.append("    @classmethod")
    lines.append(f"    def from_bytes(cls, buf: bytes, offset: int = 0) -> {st.name}:")
    lines.append("        if offset < 0 or len(buf) - offset < cls.SIZE:")
    lines.append("            raise ValueError('Buffer too small for struct unpack')")
    lines.append("        return cls(")
    for fld in st.fields:
        lines.append(f"            {fld.name}={_unpack_expr(fld)},")
    lines.append("        )")
    lines.append("")
    lines.append(f"{st.name}.SIZE = 0x{st.size:x}")
    lines.append(f"{st.name}.OFFSETS = {{")
    for fld in st.fields:
        lines.append(f"    {fld.name!r}: 0x{fld.offset:x},")
    lines.append("}")
    lines.append("")
    return lines


def _emit_py_chain(ch: PointerChain) -> list[str]:
    fsize, py_ty, unpack_code = _chain_unpack(ch.field_kind)
    offsets_lit = ", ".join(f"0x{off:x}" for off in ch.offsets)
    return [
        f"CHAIN_{ch.name}_ROOT_RVA: int = 0x{ch.root_rva:x}",
        f"CHAIN_{ch.name}_OFFSETS: list[int] = [{offsets_lit}]",
        "",
        "",
        f"def resolve_chain_{ch.name}(",
        "    read_mem: Callable[[int, int], bytes | None],",
        "    image_base: int,",
        f") -> {py_ty} | None:",
        f"    raw = read_mem(image_base + CHAIN_{ch.name}_ROOT_RVA, 8)",
        "    if raw is None or len(raw) < 8:",
        "        return None",
        "    ptr = struct.unpack('<Q', raw)[0]",
        "    if ptr == 0:",
        "        return None",
        f"    for off in CHAIN_{ch.name}_OFFSETS[:-1]:",
        "        raw = read_mem(ptr + off, 8)",
        "        if raw is None or len(raw) < 8:",
        "            return None",
        "        ptr = struct.unpack('<Q', raw)[0]",
        "        if ptr == 0:",
        "            return None",
        f"    raw = read_mem(ptr + CHAIN_{ch.name}_OFFSETS[-1], {fsize})",
        f"    if raw is None or len(raw) < {fsize}:",
        "        return None",
        f"    return {unpack_code}",
        "",
    ]


def generate_python_sdk(spec: SdkSpec) -> str:
    validate_spec(spec)
    safe_mod = spec.module_name.replace("\r", " ").replace("\n", " ").replace('"""', "'''")
    lines: list[str] = [
        f'"""Standalone Python SDK generated by structforge for {spec.name} ({safe_mod})."""',
        "from __future__ import annotations",
        "",
        "import struct",
        "from dataclasses import dataclass",
        "from typing import Callable",
        "",
        "",
        "@dataclass(frozen=True)",
        "class RawVectorView:",
        "    begin_va: int",
        "    end_va: int",
        "    cap_va: int",
        "",
        "    @property",
        "    def byte_len(self) -> int:",
        "        return max(0, self.end_va - self.begin_va)",
        "",
        "    def count(self, elem_size: int) -> int:",
        "        return self.byte_len // elem_size if elem_size > 0 else 0",
        "",
        "",
        "def _read_cstr_mem(",
        "    read_mem: Callable[[int, int], bytes | None], va: int, max_len: int = 64",
        ") -> str | None:",
        "    if va <= 0:",
        "        return None",
        "    for sz in (max_len, 32, 16, 8, 4):",
        "        if sz > max_len:",
        "            continue",
        "        chunk = read_mem(va, sz)",
        "        if chunk is not None:",
        "            nul = chunk.find(b'\\x00')",
        "            if nul >= 0:",
        "                return chunk[:nul].decode('ascii', errors='replace')",
        "    out = bytearray()",
        "    for i in range(max_len):",
        "        b = read_mem(va + i, 1)",
        "        if b is None:",
        "            return None",
        "        if b[0] == 0:",
        "            return out.decode('ascii', errors='replace')",
        "        out.append(b[0])",
        "    return None",
        "",
        "",
    ]

    for st in spec.structs:
        lines.extend(_emit_py_struct(st))
        lines.append("")

    for ch in spec.pointer_chains:
        lines.extend(_emit_py_chain(ch))

    return "\n".join(lines)
