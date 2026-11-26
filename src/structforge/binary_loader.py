from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from structforge.errors import BinaryFormatError

_MAX_SECTION_SIZE = 16 * 1024 * 1024


@dataclass(frozen=True)
class SectionSlice:
    name: str
    rva: int
    data: bytes
    is_exec: bool
    is_write: bool


@dataclass(frozen=True)
class SymbolEntry:
    name: str
    rva: int
    size: int
    is_func: bool


@dataclass
class LoadedBinary:
    format: Literal["pe64", "elf64"]
    image_base: int
    sections: list[SectionSlice]
    symbols: list[SymbolEntry] = field(default_factory=list)

    def section_at_rva(self, rva: int) -> SectionSlice | None:
        if rva < 0:
            return None
        for sec in self.sections:
            if sec.rva <= rva < sec.rva + len(sec.data):
                return sec
        return None

    def read_rva(self, rva: int, size: int) -> bytes | None:
        if size <= 0 or rva < 0:
            return None
        sec = self.section_at_rva(rva)
        if sec is None:
            return None
        off = rva - sec.rva
        if off + size > len(sec.data):
            return None
        return sec.data[off : off + size]

    def read_cstr_rva(self, rva: int, max_len: int = 256) -> str | None:
        if rva < 0 or max_len <= 0:
            return None
        sec = self.section_at_rva(rva)
        if sec is None:
            return None
        off = rva - sec.rva
        end = min(len(sec.data), off + max_len)
        chunk = sec.data[off:end]
        nul = chunk.find(b"\x00")
        if nul < 0:
            return None
        raw = chunk[:nul]
        try:
            return raw.decode("ascii")
        except UnicodeDecodeError:
            return None

    def is_exec_rva(self, rva: int) -> bool:
        sec = self.section_at_rva(rva)
        return bool(sec is not None and sec.is_exec)


def load_binary(data: bytes) -> LoadedBinary:
    if len(data) >= 4 and data[:4] == b"\x7fELF":
        return _parse_elf64(data)
    if len(data) >= 0x40 and data[:2] == b"MZ":
        return _parse_pe64(data)
    raise BinaryFormatError("Unrecognized binary format (expected PE32+ or ELF64)")


def load_binary_file(path: str | Path) -> LoadedBinary:
    return load_binary(Path(path).read_bytes())


def _parse_pe64(data: bytes) -> LoadedBinary:
    if len(data) < 0x40:
        raise BinaryFormatError("PE32+ file too small for DOS header")
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if e_lfanew + 24 > len(data) or data[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
        raise BinaryFormatError("Invalid PE signature")

    coff_off = e_lfanew + 4
    machine, num_sections, _, _, _, opt_size, _ = struct.unpack_from("<HHIIIHH", data, coff_off)
    if machine != 0x8664:
        raise BinaryFormatError(f"Unsupported PE machine 0x{machine:04x} (expected x86-64 0x8664)")
    opt_off = coff_off + 20
    if opt_size < 112 or opt_off + opt_size > len(data):
        raise BinaryFormatError("Truncated PE32+ optional header")

    magic = struct.unpack_from("<H", data, opt_off)[0]
    if magic != 0x020B:
        raise BinaryFormatError(
            f"Unsupported PE optional magic 0x{magic:04x} (expected PE32+ 0x020B)"
        )

    image_base = struct.unpack_from("<Q", data, opt_off + 24)[0]
    sec_table_off = opt_off + opt_size
    if sec_table_off + num_sections * 40 > len(data):
        raise BinaryFormatError("Truncated PE32+ section table")

    sections: list[SectionSlice] = []
    for idx in range(num_sections):
        raw_name, vsize, va_rva, rsize, roff, _, _, _, _, chars = struct.unpack_from(
            "<8sIIIIIIHHI", data, sec_table_off + idx * 40
        )
        name = raw_name.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        logical_len = vsize if vsize > 0 else rsize
        logical_len = min(logical_len, _MAX_SECTION_SIZE)
        if logical_len == 0:
            continue
        if roff < len(data) and rsize > 0:
            avail = min(rsize, len(data) - roff, logical_len)
            sec_bytes = data[roff : roff + avail]
        else:
            sec_bytes = b""
        if len(sec_bytes) < logical_len:
            sec_bytes = sec_bytes + b"\x00" * (logical_len - len(sec_bytes))
        is_exec = bool(chars & 0x20000020)
        is_write = bool(chars & 0x80000000)
        sections.append(
            SectionSlice(name=name, rva=va_rva, data=sec_bytes, is_exec=is_exec, is_write=is_write)
        )

    return LoadedBinary(format="pe64", image_base=image_base, sections=sections, symbols=[])


def _parse_elf64(data: bytes) -> LoadedBinary:
    if len(data) < 64:
        raise BinaryFormatError("ELF64 file smaller than 64-byte header")
    ei_class = data[4]
    ei_data = data[5]
    if ei_class != 2 or ei_data != 1:
        raise BinaryFormatError("Only little-endian ELF64 is supported")

    (
        _e_type,
        e_machine,
        _e_version,
        _e_entry,
        e_phoff,
        e_shoff,
        _e_flags,
        _e_ehsize,
        e_phentsize,
        e_phnum,
        e_shentsize,
        e_shnum,
        e_shstrndx,
    ) = struct.unpack_from("<HHIQQQIHHHHHH", data, 16)

    if e_machine != 62:
        raise BinaryFormatError(f"Unsupported ELF machine {e_machine} (expected EM_X86_64=62)")

    load_vaddrs: list[int] = []
    if e_phoff > 0 and e_phnum > 0:
        if e_phentsize < 56 or e_phoff + e_phnum * e_phentsize > len(data):
            raise BinaryFormatError("Truncated ELF64 program header table")
        for idx in range(e_phnum):
            p_type, _p_flags, _p_off, p_vaddr, _p_paddr, p_filesz, _p_memsz, _p_align = (
                struct.unpack_from("<IIQQQQQQ", data, e_phoff + idx * e_phentsize)
            )
            if p_type == 1 and p_filesz > 0:
                load_vaddrs.append(p_vaddr)

    image_base = min(load_vaddrs, default=0)

    if e_shoff == 0 or e_shnum == 0:
        return LoadedBinary(format="elf64", image_base=image_base, sections=[], symbols=[])

    if e_shentsize < 64 or e_shoff + e_shnum * e_shentsize > len(data):
        raise BinaryFormatError("Truncated ELF64 section header table")

    shdrs: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    for idx in range(e_shnum):
        sh = struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + idx * e_shentsize)
        shdrs.append(sh)

    shstr_data = b""
    if 0 <= e_shstrndx < len(shdrs):
        _, _, _, _, shstr_off, shstr_size, _, _, _, _ = shdrs[e_shstrndx]
        if shstr_off + shstr_size <= len(data):
            shstr_data = data[shstr_off : shstr_off + shstr_size]

    def _read_str(tbl: bytes, offset: int) -> str:
        if offset < 0 or offset >= len(tbl):
            return ""
        end = tbl.find(b"\x00", offset)
        if end < 0:
            end = len(tbl)
        return tbl[offset:end].decode("utf-8", errors="replace")

    sections: list[SectionSlice] = []
    symbols: list[SymbolEntry] = []

    for (
        sh_name,
        sh_type,
        sh_flags,
        sh_addr,
        sh_offset,
        sh_size,
        sh_link,
        _info,
        _align,
        entsize,
    ) in shdrs:
        sec_name = _read_str(shstr_data, sh_name)
        if (sh_flags & 0x2) and sh_type != 8:
            clamped_size = min(sh_size, _MAX_SECTION_SIZE)
            if sh_offset + clamped_size > len(data):
                raise BinaryFormatError(f"Truncated ELF64 section '{sec_name}'")
            sec_data = data[sh_offset : sh_offset + clamped_size]
            rva = sh_addr - image_base if sh_addr >= image_base else sh_addr
            is_exec = bool(sh_flags & 0x4)
            is_write = bool(sh_flags & 0x1)
            sections.append(
                SectionSlice(
                    name=sec_name, rva=rva, data=sec_data, is_exec=is_exec, is_write=is_write
                )
            )

        if sh_type in (2, 11) and sh_link < len(shdrs):
            _, _, _, _, str_off, str_size, _, _, _, _ = shdrs[sh_link]
            if str_off + str_size > len(data) or sh_offset + sh_size > len(data):
                raise BinaryFormatError(f"Truncated ELF64 symbol/string table '{sec_name}'")
            strtab = data[str_off : str_off + str_size]
            step = entsize if entsize >= 24 else 24
            sym_end = sh_offset + sh_size
            pos = sh_offset
            while pos + 24 <= sym_end:
                st_name, st_info, _st_other, _st_shndx, st_value, st_size = struct.unpack_from(
                    "<IBBHQQ", data, pos
                )
                pos += step
                sym_name = _read_str(strtab, st_name)
                if sym_name and st_value > 0:
                    rva = st_value - image_base if st_value >= image_base else st_value
                    symbols.append(
                        SymbolEntry(
                            name=sym_name,
                            rva=rva,
                            size=st_size,
                            is_func=((st_info & 0xF) == 2),
                        )
                    )

    return LoadedBinary(format="elf64", image_base=image_base, sections=sections, symbols=symbols)
