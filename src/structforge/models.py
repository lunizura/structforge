from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from structforge.errors import SpecValidationError

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CPP_KEYWORDS = frozenset(
    {
        "alignas",
        "alignof",
        "auto",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "constexpr",
        "continue",
        "default",
        "delete",
        "do",
        "double",
        "else",
        "enum",
        "explicit",
        "export",
        "extern",
        "false",
        "float",
        "for",
        "friend",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "mutable",
        "namespace",
        "new",
        "noexcept",
        "nullptr",
        "operator",
        "private",
        "protected",
        "public",
        "register",
        "reinterpret_cast",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "static_cast",
        "struct",
        "switch",
        "template",
        "this",
        "throw",
        "true",
        "try",
        "typedef",
        "typeid",
        "typename",
        "union",
        "unsigned",
        "using",
        "virtual",
        "void",
        "volatile",
        "while",
    }
)


def sanitize_ident(raw: str, prefix: str = "sf_") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw.replace("::", "_")).strip("_")
    if not cleaned:
        cleaned = f"{prefix}item"
    if cleaned[0].isdigit():
        cleaned = f"{prefix}{cleaned}"
    if cleaned in _CPP_KEYWORDS:
        cleaned = f"{cleaned}_"
    return cleaned


class FieldKind(str, Enum):
    VPTR = "vptr"  # 8B, align 8 (std::uint64_t)
    PTR64 = "ptr64"  # 8B, align 8 (std::uint64_t)
    CSTR_PTR = "cstr_ptr"  # 8B, align 8 (std::uint64_t -> const char*)
    STD_VECTOR = "std_vector"  # 24B, align 8 (RawVectorView {begin, end, cap})
    F64 = "f64"  # 8B, align 8 (double)
    I64 = "i64"  # 8B, align 8 (std::int64_t)
    U64 = "u64"  # 8B, align 8 (std::uint64_t)
    F32 = "f32"  # 4B, align 4 (float)
    I32 = "i32"  # 4B, align 4 (std::int32_t)
    U32 = "u32"  # 4B, align 4 (std::uint32_t)
    U16 = "u16"  # 2B, align 2 (std::uint16_t)
    U8 = "u8"  # 1B, align 1 (std::uint8_t)
    BOOL = "bool"  # 1B, align 1 (bool)
    CHAR_ARRAY = "char_array"  # NB, align 1 (char[N])
    PAD = "pad"  # NB, align 1 (std::uint8_t[N])


FIELD_ALIGNMENT: dict[FieldKind, int] = {
    FieldKind.VPTR: 8,
    FieldKind.PTR64: 8,
    FieldKind.CSTR_PTR: 8,
    FieldKind.STD_VECTOR: 8,
    FieldKind.F64: 8,
    FieldKind.I64: 8,
    FieldKind.U64: 8,
    FieldKind.F32: 4,
    FieldKind.I32: 4,
    FieldKind.U32: 4,
    FieldKind.U16: 2,
    FieldKind.U8: 1,
    FieldKind.BOOL: 1,
    FieldKind.CHAR_ARRAY: 1,
    FieldKind.PAD: 1,
}

FIXED_FIELD_SIZES: dict[FieldKind, int] = {
    FieldKind.VPTR: 8,
    FieldKind.PTR64: 8,
    FieldKind.CSTR_PTR: 8,
    FieldKind.STD_VECTOR: 24,
    FieldKind.F64: 8,
    FieldKind.I64: 8,
    FieldKind.U64: 8,
    FieldKind.F32: 4,
    FieldKind.I32: 4,
    FieldKind.U32: 4,
    FieldKind.U16: 2,
    FieldKind.U8: 1,
    FieldKind.BOOL: 1,
}


@dataclass(frozen=True)
class VTableInfo:
    mangled_name: str
    demangled_name: str
    vtable_rva: int
    method_rvas: tuple[int, ...]
    base_classes: tuple[str, ...] = ()
    subobject_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mangled_name": self.mangled_name,
            "demangled_name": self.demangled_name,
            "vtable_rva": self.vtable_rva,
            "method_rvas": list(self.method_rvas),
            "base_classes": list(self.base_classes),
            "subobject_offset": self.subobject_offset,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VTableInfo:
        return cls(
            mangled_name=str(d["mangled_name"]),
            demangled_name=str(d["demangled_name"]),
            vtable_rva=int(d["vtable_rva"]),
            method_rvas=tuple(int(x) for x in d.get("method_rvas", [])),
            base_classes=tuple(str(x) for x in d.get("base_classes", [])),
            subobject_offset=int(d.get("subobject_offset", 0)),
        )


@dataclass
class StructField:
    name: str
    offset: int
    size: int
    kind: FieldKind
    target_type: str | None = None
    vtable_rva: int | None = None
    elem_stride: int | None = None
    sample_values: list[Any] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def alignment(self) -> int:
        return FIELD_ALIGNMENT[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "offset": self.offset,
            "size": self.size,
            "kind": self.kind.value,
            "target_type": self.target_type,
            "vtable_rva": self.vtable_rva,
            "elem_stride": self.elem_stride,
            "sample_values": list(self.sample_values),
            "confidence": round(self.confidence, 3),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructField:
        return cls(
            name=str(d["name"]),
            offset=int(d["offset"]),
            size=int(d["size"]),
            kind=FieldKind(d["kind"]),
            target_type=str(d["target_type"]) if d.get("target_type") is not None else None,
            vtable_rva=int(d["vtable_rva"]) if d.get("vtable_rva") is not None else None,
            elem_stride=int(d["elem_stride"]) if d.get("elem_stride") is not None else None,
            sample_values=list(d.get("sample_values", [])),
            confidence=float(d.get("confidence", 1.0)),
        )


@dataclass
class StructDef:
    name: str
    size: int
    alignment: int
    fields: list[StructField] = field(default_factory=list)
    demangled_rtti: str | None = None
    vtable_rva: int | None = None
    base_classes: list[str] = field(default_factory=list)
    instance_vas: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "alignment": self.alignment,
            "demangled_rtti": self.demangled_rtti,
            "vtable_rva": self.vtable_rva,
            "base_classes": list(self.base_classes),
            "instance_vas": list(self.instance_vas),
            "fields": [f.to_dict() for f in self.fields],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StructDef:
        return cls(
            name=str(d["name"]),
            size=int(d["size"]),
            alignment=int(d["alignment"]),
            demangled_rtti=str(d["demangled_rtti"])
            if d.get("demangled_rtti") is not None
            else None,
            vtable_rva=int(d["vtable_rva"]) if d.get("vtable_rva") is not None else None,
            base_classes=[str(x) for x in d.get("base_classes", [])],
            instance_vas=[int(x) for x in d.get("instance_vas", [])],
            fields=[StructField.from_dict(x) for x in d.get("fields", [])],
        )


@dataclass
class PointerChain:
    name: str
    root_rva: int
    offsets: list[int]
    field_kind: FieldKind
    expected_values: list[Any] = field(default_factory=list)
    target_struct: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root_rva": self.root_rva,
            "offsets": list(self.offsets),
            "field_kind": self.field_kind.value,
            "expected_values": list(self.expected_values),
            "target_struct": self.target_struct,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PointerChain:
        return cls(
            name=str(d["name"]),
            root_rva=int(d["root_rva"]),
            offsets=[int(x) for x in d.get("offsets", [])],
            field_kind=FieldKind(d["field_kind"]),
            expected_values=list(d.get("expected_values", [])),
            target_struct=str(d["target_struct"]) if d.get("target_struct") is not None else None,
        )


@dataclass
class SdkSpec:
    name: str
    binary_format: Literal["pe64", "elf64", "raw64"] = "pe64"
    module_name: str = "module.dll"
    structs: list[StructDef] = field(default_factory=list)
    vtables: list[VTableInfo] = field(default_factory=list)
    pointer_chains: list[PointerChain] = field(default_factory=list)
    layout_score: float = 0.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "binary_format": self.binary_format,
            "module_name": self.module_name,
            "layout_score": round(self.layout_score, 2),
            "description": self.description,
            "vtables": [v.to_dict() for v in self.vtables],
            "structs": [s.to_dict() for s in self.structs],
            "pointer_chains": [c.to_dict() for c in self.pointer_chains],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SdkSpec:
        if not isinstance(d, dict):
            raise SpecValidationError("SdkSpec root must be a JSON object")
        spec = cls(
            name=str(d.get("name", "")),
            binary_format=d.get("binary_format", "pe64"),
            module_name=str(d.get("module_name", "module.dll")),
            layout_score=float(d.get("layout_score", 0.0)),
            description=str(d.get("description", "")),
            vtables=[VTableInfo.from_dict(x) for x in d.get("vtables", [])],
            structs=[StructDef.from_dict(x) for x in d.get("structs", [])],
            pointer_chains=[PointerChain.from_dict(x) for x in d.get("pointer_chains", [])],
        )
        validate_spec(spec)
        return spec

    def to_json(self) -> str:
        validate_spec(self)
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> SdkSpec:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SpecValidationError(f"Invalid JSON in SdkSpec: {exc}") from exc
        return cls.from_dict(raw)


def validate_spec(spec: SdkSpec) -> None:
    if not _IDENT_RE.match(spec.name) or spec.name in _CPP_KEYWORDS:
        raise SpecValidationError(f"Invalid spec identifier '{spec.name}'")
    if spec.binary_format not in ("pe64", "elf64", "raw64"):
        raise SpecValidationError(f"Unsupported binary_format '{spec.binary_format}'")
    if not spec.structs:
        raise SpecValidationError("SdkSpec must contain at least one struct")

    seen_structs: set[str] = set()
    for st in spec.structs:
        if not _IDENT_RE.match(st.name) or st.name in _CPP_KEYWORDS:
            raise SpecValidationError(f"Invalid struct identifier '{st.name}'")
        if st.name in seen_structs:
            raise SpecValidationError(f"Duplicate struct name '{st.name}'")
        seen_structs.add(st.name)
        if st.size <= 0 or st.alignment not in (1, 2, 4, 8):
            raise SpecValidationError(f"Invalid size/alignment in struct '{st.name}'")
        if not st.fields:
            raise SpecValidationError(f"Struct '{st.name}' must have at least one field")

        seen_fields: set[str] = set()
        for fld in st.fields:
            if not _IDENT_RE.match(fld.name) or fld.name in _CPP_KEYWORDS:
                raise SpecValidationError(f"Invalid field identifier '{fld.name}' in '{st.name}'")
            if fld.name in seen_fields:
                raise SpecValidationError(f"Duplicate field '{fld.name}' in '{st.name}'")
            seen_fields.add(fld.name)
            if fld.size <= 0 or fld.offset < 0:
                raise SpecValidationError(f"Invalid offset/size on '{st.name}.{fld.name}'")
            if fld.kind in FIXED_FIELD_SIZES and fld.size != FIXED_FIELD_SIZES[fld.kind]:
                raise SpecValidationError(
                    f"Field '{st.name}.{fld.name}' kind '{fld.kind.value}' requires size {FIXED_FIELD_SIZES[fld.kind]}"
                )

    seen_chains: set[str] = set()
    for ch in spec.pointer_chains:
        if not _IDENT_RE.match(ch.name) or ch.name in _CPP_KEYWORDS:
            raise SpecValidationError(f"Invalid chain identifier '{ch.name}'")
        if ch.name in seen_chains:
            raise SpecValidationError(f"Duplicate chain name '{ch.name}'")
        seen_chains.add(ch.name)
        if ch.root_rva < 0 or not ch.offsets:
            raise SpecValidationError(f"Invalid root_rva/offsets in chain '{ch.name}'")
        if ch.field_kind not in FIXED_FIELD_SIZES:
            raise SpecValidationError(
                f"PointerChain '{ch.name}' requires fixed-size scalar kind, got '{ch.field_kind.value}'"
            )
