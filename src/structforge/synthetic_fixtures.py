from __future__ import annotations

import struct
from dataclasses import dataclass

from structforge.binary_loader import LoadedBinary, load_binary
from structforge.rtti import harvest_rtti
from structforge.snapshot import HeapSnapshot, MemoryRegion


@dataclass
class DemoSuite:
    pe_bytes: bytes
    elf_bytes: bytes
    pe_binary: LoadedBinary
    elf_binary: LoadedBinary
    snapshots: list[HeapSnapshot]
    vtable_rvas: dict[str, int]
    player_vas: list[int]
    session_vas: list[int]
    world_vas: list[int]


def build_synthetic_pe_rtti(image_base: int = 0x140000000) -> tuple[bytes, dict[str, int]]:
    text_rva = 0x1000
    text_size = 0x1000
    rdata_rva = 0x2000
    rdata_size = 0x1000
    data_rva = 0x5000
    data_size = 0x1000

    text_buf = bytearray(text_size)
    for stub_idx in range(12):
        # x86-64 stub: mov eax, stub_idx; ret
        off = stub_idx * 0x10
        text_buf[off : off + 6] = b"\xb8" + struct.pack("<I", stub_idx) + b"\xc3"

    rdata_buf = bytearray(rdata_size)
    cursor = 0

    def _alloc(size: int, align: int = 8) -> int:
        nonlocal cursor
        cursor = (cursor + (align - 1)) & ~(align - 1)
        pos = cursor
        cursor += size
        return pos

    def _put_td(mangled: str) -> int:
        raw = mangled.encode("ascii") + b"\x00"
        off = _alloc(16 + len(raw), align=8)
        # pVFTable (0), spare (0), name[0...] at +16
        rdata_buf[off + 16 : off + 16 + len(raw)] = raw
        return rdata_rva + off

    def _put_bcd(td_rva: int, num_contained: int = 0) -> int:
        off = _alloc(28, align=4)
        struct.pack_into("<IIIIIII", rdata_buf, off, td_rva, num_contained, 0, 0xFFFFFFFF, 0, 0, 0)
        return rdata_rva + off

    def _put_bca(bcd_rvas: list[int]) -> int:
        off = _alloc(4 * len(bcd_rvas), align=4)
        for i, rva in enumerate(bcd_rvas):
            struct.pack_into("<I", rdata_buf, off + i * 4, rva)
        return rdata_rva + off

    def _put_chd(bca_rva: int, num_bases: int) -> int:
        off = _alloc(16, align=4)
        struct.pack_into("<IIII", rdata_buf, off, 0, 0, num_bases, bca_rva)
        return rdata_rva + off

    def _put_col(td_rva: int, chd_rva: int, sub_off: int = 0) -> int:
        off = _alloc(24, align=4)
        col_rva = rdata_rva + off
        struct.pack_into("<IIIIII", rdata_buf, off, 1, sub_off, 0, td_rva, chd_rva, col_rva)
        return col_rva

    def _put_vtable(col_rva: int, fn_rvas: list[int]) -> int:
        off = _alloc(8 * (1 + len(fn_rvas) + 1), align=8)
        struct.pack_into("<Q", rdata_buf, off, image_base + col_rva)
        for i, fn_rva in enumerate(fn_rvas):
            struct.pack_into("<Q", rdata_buf, off + 8 + i * 8, image_base + fn_rva)
        # Trailing 0 sentinel
        struct.pack_into("<Q", rdata_buf, off + 8 + len(fn_rvas) * 8, 0)
        return rdata_rva + off + 8

    td_entity = _put_td(".?AVEntity@engine@@")
    td_player = _put_td(".?AVPlayerEntity@game@@")
    td_session = _put_td(".?AVSession@game@@")

    bcd_entity = _put_bcd(td_entity, 0)
    bcd_player = _put_bcd(td_player, 1)
    bcd_session = _put_bcd(td_session, 0)

    bca_entity = _put_bca([bcd_entity])
    bca_player = _put_bca([bcd_player, bcd_entity])
    bca_session = _put_bca([bcd_session])

    chd_entity = _put_chd(bca_entity, 1)
    chd_player = _put_chd(bca_player, 2)
    chd_session = _put_chd(bca_session, 1)

    col_entity = _put_col(td_entity, chd_entity)
    col_player = _put_col(td_player, chd_player)
    col_session = _put_col(td_session, chd_session)

    vt_entity = _put_vtable(col_entity, [text_rva + 0x00, text_rva + 0x10])
    vt_player = _put_vtable(col_player, [text_rva + 0x20, text_rva + 0x30, text_rva + 0x40])
    vt_session = _put_vtable(col_session, [text_rva + 0x50, text_rva + 0x60])

    data_buf = bytearray(data_size)

    # Assemble PE32+ image
    dos_header = bytearray(0x80)
    dos_header[0:2] = b"MZ"
    struct.pack_into("<I", dos_header, 0x3C, 0x80)

    num_sections = 3
    opt_size = 240
    coff = struct.pack("<4sHHIIIHH", b"PE\x00\x00", 0x8664, num_sections, 0, 0, 0, opt_size, 0x2022)

    opt = bytearray(opt_size)
    struct.pack_into("<HBB", opt, 0, 0x020B, 14, 0)
    struct.pack_into("<III", opt, 4, text_size, rdata_size + data_size, 0)
    struct.pack_into("<II", opt, 16, text_rva, text_rva)
    struct.pack_into("<QII", opt, 24, image_base, 0x1000, 0x200)
    struct.pack_into("<HHHHHH", opt, 40, 10, 0, 0, 0, 10, 0)
    struct.pack_into("<IIII", opt, 56, 0x6000, 0x400, 0, 3)
    struct.pack_into("<HH", opt, 68, 0x8160, 0)
    struct.pack_into("<QQQQII", opt, 72, 0x100000, 0x1000, 0x100000, 0x1000, 0, 16)

    text_raw_off = 0x400
    rdata_raw_off = text_raw_off + text_size
    data_raw_off = rdata_raw_off + rdata_size

    sec_headers = bytearray()
    sec_headers += struct.pack(
        "<8sIIIIIIHHI",
        b".text",
        text_size,
        text_rva,
        text_size,
        text_raw_off,
        0,
        0,
        0,
        0,
        0x60000020,
    )
    sec_headers += struct.pack(
        "<8sIIIIIIHHI",
        b".rdata",
        rdata_size,
        rdata_rva,
        rdata_size,
        rdata_raw_off,
        0,
        0,
        0,
        0,
        0x40000040,
    )
    sec_headers += struct.pack(
        "<8sIIIIIIHHI",
        b".data",
        data_size,
        data_rva,
        data_size,
        data_raw_off,
        0,
        0,
        0,
        0,
        0xC0000040,
    )

    headers = bytes(dos_header) + coff + bytes(opt) + bytes(sec_headers)
    headers = headers + b"\x00" * (text_raw_off - len(headers))
    pe_bytes = headers + bytes(text_buf) + bytes(rdata_buf) + bytes(data_buf)
    return pe_bytes, {
        "engine::Entity": vt_entity,
        "game::PlayerEntity": vt_player,
        "game::Session": vt_session,
    }


def build_synthetic_elf_rtti(image_base: int = 0x400000) -> bytes:
    text_rva = 0x1000
    text_size = 0x200
    rodata_rva = 0x2000
    rodata_size = 0x400

    text_buf = bytearray(text_size)
    for stub_idx in range(8):
        off = stub_idx * 0x10
        text_buf[off : off + 6] = b"\xb8" + struct.pack("<I", stub_idx) + b"\xc3"

    rodata_buf = bytearray(rodata_size)

    # Typeinfo & VTables in .rodata
    zti_entity_rva = rodata_rva + 0x00
    zti_entity_size = 16
    # __class_type_info: vptr + __type_info_name
    struct.pack_into("<QQ", rodata_buf, 0x00, 0, image_base + rodata_rva + 0x200)

    zti_player_rva = rodata_rva + 0x20
    zti_player_size = 24
    # __si_class_type_info: vptr + __type_info_name + __base_type
    struct.pack_into(
        "<QQQ", rodata_buf, 0x20, 0, image_base + rodata_rva + 0x220, image_base + zti_entity_rva
    )

    ztv_entity_rva = rodata_rva + 0x40
    ztv_entity_size = 32  # 16B header + 2 methods
    struct.pack_into(
        "<qQQQ",
        rodata_buf,
        0x40,
        0,
        image_base + zti_entity_rva,
        image_base + text_rva + 0x00,
        image_base + text_rva + 0x10,
    )

    ztv_player_rva = rodata_rva + 0x70
    ztv_player_size = 40  # 16B header + 3 methods
    struct.pack_into(
        "<qQQQQ",
        rodata_buf,
        0x70,
        0,
        image_base + zti_player_rva,
        image_base + text_rva + 0x20,
        image_base + text_rva + 0x30,
        image_base + text_rva + 0x40,
    )

    shstr_items = [b"", b".text", b".rodata", b".symtab", b".shstrtab", b".strtab"]
    shstr_buf = bytearray()
    shstr_offs: list[int] = []
    for item in shstr_items:
        shstr_offs.append(len(shstr_buf))
        shstr_buf.extend(item + b"\x00")

    str_buf = bytearray(b"\x00")

    def _add_str(s: str) -> int:
        pos = len(str_buf)
        str_buf.extend(s.encode("ascii") + b"\x00")
        return pos

    sym_entries: list[tuple[int, int, int, int, int, int]] = [
        (0, 0, 0, 0, 0, 0),
        (_add_str("_ZTI6Entity"), 0x11, 0, 2, image_base + zti_entity_rva, zti_entity_size),
        (
            _add_str("_ZTIN4game12PlayerEntityE"),
            0x11,
            0,
            2,
            image_base + zti_player_rva,
            zti_player_size,
        ),
        (_add_str("_ZTV6Entity"), 0x11, 0, 2, image_base + ztv_entity_rva, ztv_entity_size),
        (
            _add_str("_ZTVN4game12PlayerEntityE"),
            0x11,
            0,
            2,
            image_base + ztv_player_rva,
            ztv_player_size,
        ),
        (_add_str("_ZN6Entity4tickEv"), 0x12, 0, 1, image_base + text_rva + 0x00, 0x10),
        (
            _add_str("_ZN4game12PlayerEntity4tickEv"),
            0x12,
            0,
            1,
            image_base + text_rva + 0x20,
            0x10,
        ),
    ]
    symtab_buf = bytearray()
    for st_name, st_info, st_other, st_shndx, st_value, st_size in sym_entries:
        symtab_buf.extend(
            struct.pack("<IBBHQQ", st_name, st_info, st_other, st_shndx, st_value, st_size)
        )

    ehdr_size = 64
    phdr_size = 56
    text_off = 0x200
    rodata_off = text_off + text_size
    symtab_off = rodata_off + rodata_size
    shstr_off = symtab_off + len(symtab_buf)
    str_off = shstr_off + len(shstr_buf)
    shoff = (str_off + len(str_buf) + 7) & ~7

    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01\x01" + b"\x00" * 9,
        3,  # ET_DYN
        62,  # EM_X86_64
        1,
        image_base + text_rva,
        ehdr_size,
        shoff,
        0,
        ehdr_size,
        phdr_size,
        1,
        64,
        6,
        4,
    )

    phdr = struct.pack(
        "<IIQQQQQQ",
        1,  # PT_LOAD
        5,  # PF_R | PF_X
        0,
        image_base,
        image_base,
        shoff,
        shoff,
        0x1000,
    )

    out = bytearray(ehdr + phdr)
    out.extend(b"\x00" * (text_off - len(out)))
    out.extend(text_buf)
    out.extend(rodata_buf)
    out.extend(symtab_buf)
    out.extend(shstr_buf)
    out.extend(str_buf)
    out.extend(b"\x00" * (shoff - len(out)))

    shdrs = [
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (shstr_offs[1], 1, 6, image_base + text_rva, text_off, text_size, 0, 0, 16, 0),
        (shstr_offs[2], 1, 2, image_base + rodata_rva, rodata_off, rodata_size, 0, 0, 8, 0),
        (shstr_offs[3], 2, 0, 0, symtab_off, len(symtab_buf), 5, 1, 8, 24),
        (shstr_offs[4], 3, 0, 0, shstr_off, len(shstr_buf), 0, 0, 1, 0),
        (shstr_offs[5], 3, 0, 0, str_off, len(str_buf), 0, 0, 1, 0),
    ]
    for sh in shdrs:
        out.extend(struct.pack("<IIQQQQIIQQ", *sh))

    return bytes(out)


def build_demo_snapshots(
    vtable_rvas: dict[str, int], image_base: int = 0x140000000
) -> tuple[list[HeapSnapshot], list[int], list[int], list[int]]:
    root_va = image_base + 0x5000
    snapshots: list[HeapSnapshot] = []
    world_vas: list[int] = []
    session_vas: list[int] = []
    player_vas: list[int] = []

    states = [
        # (health, max_health, ammo, is_alive, vec_items, pos_z)
        (100.0, 100.0, 30, 1, [11, 22, 33], 142.5000001),
        (72.5, 100.0, 29, 1, [11, 22, 33, 44], 143.2500001),
        (0.0, 100.0, 0, 0, [11, 22, 33, 44, 55], 140.7500001),
    ]

    for idx, (health, max_health, ammo, is_alive, vec_items, pos_z) in enumerate(states):
        heap_base = 0x20000000 + idx * 0x100000
        heap_buf = bytearray(0x1000)

        world_off = 0x100
        session_off = 0x400
        player_off = 0x700
        cstr_off = 0xA00
        vec_off = 0xD00

        world_va = heap_base + world_off
        session_va = heap_base + session_off
        player_va = heap_base + player_off
        cstr_va = heap_base + cstr_off
        vec_va = heap_base + vec_off

        world_vas.append(world_va)
        session_vas.append(session_va)
        player_vas.append(player_va)

        # Global .data region containing root_va -> world_va
        data_buf = bytearray(0x100)
        struct.pack_into("<Q", data_buf, 0, world_va)

        # WorldContext (0x20B): session_ptr @ +0x18
        struct.pack_into("<Q", heap_buf, world_off + 0x18, session_va)

        # game::Session (0x48B): vptr @ +0x00, char[16] "Fontaine_Arena" @ +0x08, local_player @ +0x40
        struct.pack_into(
            "<Q", heap_buf, session_off + 0x00, image_base + vtable_rvas["game::Session"]
        )
        arena = b"Fontaine_Arena"
        heap_buf[session_off + 0x08 : session_off + 0x08 + len(arena)] = arena
        struct.pack_into("<Q", heap_buf, session_off + 0x40, player_va)

        # Hero C-string
        hero_bytes = b"ACTUAL_HERO_01\x00"
        heap_buf[cstr_off : cstr_off + len(hero_bytes)] = hero_bytes

        # std::vector<uint32_t> buffer (cap = 32 bytes)
        for v_idx, val in enumerate(vec_items):
            struct.pack_into("<I", heap_buf, vec_off + v_idx * 4, val)
        vec_end_va = vec_va + len(vec_items) * 4
        vec_cap_va = vec_va + 32

        # game::PlayerEntity (0x48B)
        struct.pack_into(
            "<Q", heap_buf, player_off + 0x00, image_base + vtable_rvas["game::PlayerEntity"]
        )
        struct.pack_into("<i", heap_buf, player_off + 0x08, 1001)
        struct.pack_into("<i", heap_buf, player_off + 0x0C, 7)
        struct.pack_into("<f", heap_buf, player_off + 0x10, health)
        struct.pack_into("<f", heap_buf, player_off + 0x14, max_health)
        struct.pack_into("<i", heap_buf, player_off + 0x18, ammo)
        heap_buf[player_off + 0x1C] = is_alive
        # +0x1D..+0x20 is already 0x00 padding
        struct.pack_into("<Q", heap_buf, player_off + 0x20, cstr_va)
        struct.pack_into("<QQQ", heap_buf, player_off + 0x28, vec_va, vec_end_va, vec_cap_va)
        struct.pack_into("<d", heap_buf, player_off + 0x40, pos_z)

        snapshots.append(
            HeapSnapshot(
                image_base=image_base,
                root_va=root_va,
                regions=[
                    MemoryRegion(name=".data", base_va=root_va, data=bytes(data_buf), flags=2),
                    MemoryRegion(name="heap", base_va=heap_base, data=bytes(heap_buf), flags=1),
                ],
            )
        )

    return snapshots, world_vas, session_vas, player_vas


def make_demo_suite() -> DemoSuite:
    pe_bytes, vt_map = build_synthetic_pe_rtti()
    elf_bytes = build_synthetic_elf_rtti()
    pe_binary = load_binary(pe_bytes)
    elf_binary = load_binary(elf_bytes)
    pe_vts = harvest_rtti(pe_binary)
    vtable_rvas = {vt.demangled_name: vt.vtable_rva for vt in pe_vts} or vt_map
    snapshots, world_vas, session_vas, player_vas = build_demo_snapshots(vtable_rvas)
    return DemoSuite(
        pe_bytes=pe_bytes,
        elf_bytes=elf_bytes,
        pe_binary=pe_binary,
        elf_binary=elf_binary,
        snapshots=snapshots,
        vtable_rvas=vtable_rvas,
        player_vas=player_vas,
        session_vas=session_vas,
        world_vas=world_vas,
    )
