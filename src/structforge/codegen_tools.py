from __future__ import annotations

import html

from structforge.models import FieldKind, SdkSpec, StructField, validate_spec


def _esc(val: str) -> str:
    return html.escape(val, quote=True)


def _reclass_node_type(kind: FieldKind) -> str:
    return {
        FieldKind.VPTR: "VTableNode",
        FieldKind.PTR64: "PointerNode",
        FieldKind.CSTR_PTR: "UTF8TextPtrNode",
        FieldKind.STD_VECTOR: "Hex64Node",
        FieldKind.F64: "DoubleNode",
        FieldKind.I64: "Int64Node",
        FieldKind.U64: "UInt64Node",
        FieldKind.F32: "FloatNode",
        FieldKind.I32: "Int32Node",
        FieldKind.U32: "UInt32Node",
        FieldKind.U16: "UInt16Node",
        FieldKind.U8: "UInt8Node",
        FieldKind.BOOL: "BoolNode",
        FieldKind.CHAR_ARRAY: "UTF8TextNode",
        FieldKind.PAD: "Hex8Node",
    }[kind]


def _ce_var_type(kind: FieldKind) -> str:
    return {
        FieldKind.VPTR: "8 Bytes",
        FieldKind.PTR64: "8 Bytes",
        FieldKind.CSTR_PTR: "8 Bytes",
        FieldKind.STD_VECTOR: "Array of byte",
        FieldKind.F64: "Double",
        FieldKind.I64: "8 Bytes",
        FieldKind.U64: "8 Bytes",
        FieldKind.F32: "Float",
        FieldKind.I32: "4 Bytes",
        FieldKind.U32: "4 Bytes",
        FieldKind.U16: "2 Bytes",
        FieldKind.U8: "Byte",
        FieldKind.BOOL: "Byte",
        FieldKind.CHAR_ARRAY: "String",
        FieldKind.PAD: "Array of byte",
    }[kind]


def _ce_struct_vartype(kind: FieldKind) -> str:
    return {
        FieldKind.VPTR: "Pointer",
        FieldKind.PTR64: "Pointer",
        FieldKind.CSTR_PTR: "Pointer",
        FieldKind.STD_VECTOR: "Array of byte",
        FieldKind.F64: "Double",
        FieldKind.I64: "8 Bytes",
        FieldKind.U64: "8 Bytes",
        FieldKind.F32: "Float",
        FieldKind.I32: "4 Bytes",
        FieldKind.U32: "4 Bytes",
        FieldKind.U16: "2 Bytes",
        FieldKind.U8: "Byte",
        FieldKind.BOOL: "Byte",
        FieldKind.CHAR_ARRAY: "String",
        FieldKind.PAD: "Array of byte",
    }[kind]


def _emit_reclass_nodes(fld: StructField) -> list[str]:
    if fld.kind == FieldKind.STD_VECTOR:
        name_esc = _esc(fld.name)
        return [
            f'      <Node Name="{name_esc}_begin" Type="PointerNode" Comment="std_vector begin" />',
            f'      <Node Name="{name_esc}_end" Type="PointerNode" Comment="std_vector end" />',
            f'      <Node Name="{name_esc}_cap" Type="PointerNode" Comment="std_vector cap" />',
        ]
    if fld.kind == FieldKind.PAD:
        return [
            f'      <Node Name="{_esc(fld.name)}_{i}" Type="Hex8Node" />' for i in range(fld.size)
        ]
    node_type = _reclass_node_type(fld.kind)
    attrs = [f'Name="{_esc(fld.name)}"', f'Type="{node_type}"']
    if fld.kind == FieldKind.CHAR_ARRAY:
        attrs.append(f'Length="{fld.size}"')
    if fld.target_type:
        attrs.append(f'Reference="{_esc(fld.target_type)}"')
    return [f"      <Node {' '.join(attrs)} />"]


def generate_reclass_xml(spec: SdkSpec) -> str:
    validate_spec(spec)
    lines: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f'<ReClassNet Platform="x64" Name="{_esc(spec.name)}">',
        "  <Classes>",
    ]
    for st in spec.structs:
        comment = _esc(st.demangled_rtti or "")
        lines.append(f'    <Class Name="{_esc(st.name)}" Size="{st.size}" Comment="{comment}">')
        for fld in st.fields:
            lines.extend(_emit_reclass_nodes(fld))
        lines.append("    </Class>")
    lines.append("  </Classes>")
    lines.append("</ReClassNet>")
    lines.append("")
    return "\n".join(lines)


def generate_cheat_engine_xml(spec: SdkSpec) -> str:
    validate_spec(spec)
    mod_esc = _esc(spec.module_name)
    lines: list[str] = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<CheatTable CheatEngineTableVersion="45">',
        "  <CheatEntries>",
    ]
    for idx, ch in enumerate(spec.pointer_chains):
        lines.append("    <CheatEntry>")
        lines.append(f"      <ID>{idx}</ID>")
        lines.append(f'      <Description>"{_esc(ch.name)}"</Description>')
        lines.append(f"      <VariableType>{_ce_var_type(ch.field_kind)}</VariableType>")
        lines.append(f"      <Address>{mod_esc}+{ch.root_rva:X}</Address>")
        lines.append("      <Offsets>")
        # Cheat Engine requires leaf offset first and first-deref offset last
        for off in reversed(ch.offsets):
            lines.append(f"        <Offset>{off:X}</Offset>")
        lines.append("      </Offsets>")
        lines.append("    </CheatEntry>")
    lines.append("  </CheatEntries>")
    lines.append("  <Structures>")
    for st in spec.structs:
        lines.append(
            f'    <Structure Name="{_esc(st.name)}" AutoFill="0" AutoCreate="1" '
            f'DefaultHex="0" AutoDestroy="0" DoNotSaveLocal="0" RLECompression="1" '
            f'AutoCreateStructsize="{st.size}">'
        )
        lines.append("      <Elements>")
        for fld in st.fields:
            lines.append(
                f'        <Element Offset="{fld.offset}" Vartype="{_ce_struct_vartype(fld.kind)}" '
                f'Bytesize="{fld.size}" Description="{_esc(fld.name)}"/>'
            )
        lines.append("      </Elements>")
        lines.append("    </Structure>")
    lines.append("  </Structures>")
    lines.append("</CheatTable>")
    lines.append("")
    return "\n".join(lines)
