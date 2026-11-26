from __future__ import annotations

import struct

from structforge.binary_loader import LoadedBinary
from structforge.models import VTableInfo


def demangle_msvc_rtti(mangled: str) -> str:
    s = mangled
    if s.startswith((".?AV", ".?AU")):
        s = s[4:]
    if s.endswith("@@"):
        s = s[:-2]
    parts = [p for p in s.split("@") if p]
    if not parts:
        return mangled
    parts.reverse()
    return "::".join(parts)


def demangle_itanium_type(mangled: str) -> str:
    s = mangled
    for prefix in ("_ZTV", "_ZTI", "_ZTS"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break

    def _parse_len_name(buf: str, pos: int) -> tuple[str, int] | None:
        if pos >= len(buf) or not buf[pos].isdigit():
            return None
        num_start = pos
        while pos < len(buf) and buf[pos].isdigit():
            pos += 1
        length = int(buf[num_start:pos])
        if length <= 0 or pos + length > len(buf):
            return None
        return buf[pos : pos + length], pos + length

    if s.startswith("N") and s.endswith("E"):
        inner = s[1:-1]
        parts: list[str] = []
        pos = 0
        while pos < len(inner):
            parsed = _parse_len_name(inner, pos)
            if parsed is None:
                return mangled
            token, pos = parsed
            parts.append(token)
        return "::".join(parts) if parts else mangled

    parsed = _parse_len_name(s, 0)
    if parsed is not None:
        return parsed[0]
    return mangled


def harvest_rtti(binary: LoadedBinary) -> list[VTableInfo]:
    if binary.format == "pe64":
        vtables = _harvest_msvc_rtti(binary)
    elif binary.format == "elf64":
        vtables = _harvest_itanium_rtti(binary)
    else:
        vtables = []
    vtables.sort(key=lambda v: v.vtable_rva)
    return vtables


def _harvest_msvc_rtti(binary: LoadedBinary) -> list[VTableInfo]:
    # Step 1: Scan non-executable sections for COL_SIG_REV1 (24B, 4-byte aligned)
    cols: dict[int, tuple[int, str, str, tuple[str, ...], int]] = {}
    for sec in binary.sections:
        if sec.is_exec or len(sec.data) < 24:
            continue
        limit = len(sec.data) - 24
        for off in range(0, limit + 1, 4):
            sig, sub_off, _cd_off, td_rva, chd_rva, self_rva = struct.unpack_from(
                "<IIIIII", sec.data, off
            )
            col_rva = sec.rva + off
            if sig != 1 or self_rva != col_rva:
                continue
            td_name = binary.read_cstr_rva(td_rva + 16)
            if td_name is None:
                continue
            if not (td_name.startswith((".?AV", ".?AU")) and td_name.endswith("@@")):
                continue

            # Step 2: Parse Base Classes from CHD (chd_rva)
            base_classes: list[str] = []
            chd_bytes = binary.read_rva(chd_rva, 16)
            if chd_bytes is not None:
                chd_sig, _chd_attr, num_bases, bca_rva = struct.unpack("<IIII", chd_bytes)
                if chd_sig == 0 and 1 <= num_bases <= 64:
                    bca_bytes = binary.read_rva(bca_rva, num_bases * 4)
                    if bca_bytes is not None:
                        for b_idx in range(1, num_bases):
                            bcd_rva = struct.unpack_from("<I", bca_bytes, b_idx * 4)[0]
                            bcd_td_bytes = binary.read_rva(bcd_rva, 4)
                            if bcd_td_bytes is None:
                                continue
                            bcd_td_rva = struct.unpack("<I", bcd_td_bytes)[0]
                            base_td_name = binary.read_cstr_rva(bcd_td_rva + 16)
                            if (
                                base_td_name is not None
                                and base_td_name.startswith((".?AV", ".?AU"))
                                and base_td_name.endswith("@@")
                            ):
                                base_classes.append(demangle_msvc_rtti(base_td_name))

            col_va = binary.image_base + col_rva
            cols[col_va] = (
                col_rva,
                td_name,
                demangle_msvc_rtti(td_name),
                tuple(base_classes),
                sub_off,
            )

    if not cols:
        return []

    # Step 3: Match meta_va -> vftable in non-executable sections (8-byte aligned)
    results: list[VTableInfo] = []
    for sec in binary.sections:
        if sec.is_exec or len(sec.data) < 16:
            continue
        limit = len(sec.data) - 8
        for off in range(0, limit + 1, 8):
            col_va = struct.unpack_from("<Q", sec.data, off)[0]
            info = cols.get(col_va)
            if info is None:
                continue
            _col_rva, td_name, demangled, base_classes, sub_off = info
            vtable_rva = sec.rva + off + 8
            method_rvas: list[int] = []
            slot = 0
            while True:
                pos = off + 8 + slot * 8
                if pos + 8 > len(sec.data):
                    break
                fn_va = struct.unpack_from("<Q", sec.data, pos)[0]
                if fn_va in cols or fn_va < binary.image_base:
                    break
                fn_rva = fn_va - binary.image_base
                if not binary.is_exec_rva(fn_rva):
                    break
                method_rvas.append(fn_rva)
                slot += 1

            if method_rvas:
                results.append(
                    VTableInfo(
                        mangled_name=td_name,
                        demangled_name=demangled,
                        vtable_rva=vtable_rva,
                        method_rvas=tuple(method_rvas),
                        base_classes=base_classes,
                        subobject_offset=sub_off,
                    )
                )

    return results


def _harvest_itanium_rtti(binary: LoadedBinary) -> list[VTableInfo]:
    zti_syms = [s for s in binary.symbols if s.name.startswith("_ZTI")]
    zti_by_va: dict[int, str] = {}
    for sym in zti_syms:
        zti_by_va[binary.image_base + sym.rva] = demangle_itanium_type(sym.name)

    base_classes_by_name: dict[str, tuple[str, ...]] = {}
    for sym in zti_syms:
        if sym.size >= 24:
            base_ptr_bytes = binary.read_rva(sym.rva + 16, 8)
            if base_ptr_bytes is not None:
                base_va = struct.unpack("<Q", base_ptr_bytes)[0]
                if base_va in zti_by_va:
                    demangled = demangle_itanium_type(sym.name)
                    base_classes_by_name[demangled] = (zti_by_va[base_va],)

    results: list[VTableInfo] = []
    for sym in binary.symbols:
        if not sym.name.startswith("_ZTV") or sym.size < 24:
            continue
        hdr = binary.read_rva(sym.rva, 16)
        if hdr is None:
            continue
        offset_to_top, zti_va = struct.unpack("<qQ", hdr)
        demangled = zti_by_va.get(zti_va, demangle_itanium_type(sym.name))
        vtable_rva = sym.rva + 16

        method_rvas: list[int] = []
        slot_off = 16
        while slot_off + 8 <= sym.size:
            slot_bytes = binary.read_rva(sym.rva + slot_off, 8)
            if slot_bytes is None:
                break
            fn_va = struct.unpack("<Q", slot_bytes)[0]
            fn_rva = (
                fn_va - binary.image_base
                if binary.image_base > 0 and fn_va >= binary.image_base
                else fn_va
            )
            if binary.is_exec_rva(fn_rva):
                method_rvas.append(fn_rva)
            slot_off += 8

        if method_rvas:
            results.append(
                VTableInfo(
                    mangled_name=sym.name,
                    demangled_name=demangled,
                    vtable_rva=vtable_rva,
                    method_rvas=tuple(method_rvas),
                    base_classes=base_classes_by_name.get(demangled, ()),
                    subobject_offset=-offset_to_top if offset_to_top < 0 else offset_to_top,
                )
            )

    return results
