from __future__ import annotations

import struct

import pytest

from structforge.binary_loader import load_binary
from structforge.errors import BinaryFormatError, SnapshotFormatError, SpecValidationError
from structforge.models import FieldKind, SdkSpec, StructDef, StructField
from structforge.rtti import demangle_itanium_type, demangle_msvc_rtti, harvest_rtti
from structforge.snapshot import HeapSnapshot, MemoryRegion, parse_snapshot, serialize_snapshot
from structforge.synthetic_fixtures import make_demo_suite


def test_msvc_and_itanium_demanglers() -> None:
    assert demangle_msvc_rtti(".?AVPlayerEntity@game@@") == "game::PlayerEntity"
    assert demangle_msvc_rtti(".?AUEntity@engine@@") == "engine::Entity"
    assert demangle_msvc_rtti(".?AVSession@@") == "Session"

    assert demangle_itanium_type("_ZTVN4game12PlayerEntityE") == "game::PlayerEntity"
    assert demangle_itanium_type("_ZTIN4game12PlayerEntityE") == "game::PlayerEntity"
    assert demangle_itanium_type("_ZTV6Entity") == "Entity"


def test_harvest_pe64_and_elf64_rtti() -> None:
    suite = make_demo_suite()

    pe_vts = harvest_rtti(suite.pe_binary)
    pe_by_name = {vt.demangled_name: vt for vt in pe_vts}
    assert set(pe_by_name) == {"engine::Entity", "game::PlayerEntity", "game::Session"}
    assert pe_by_name["game::PlayerEntity"].base_classes == ("engine::Entity",)
    assert len(pe_by_name["game::PlayerEntity"].method_rvas) == 3
    assert pe_by_name["engine::Entity"].base_classes == ()

    elf_vts = harvest_rtti(suite.elf_binary)
    elf_by_name = {vt.demangled_name: vt for vt in elf_vts}
    assert set(elf_by_name) == {"Entity", "game::PlayerEntity"}
    assert elf_by_name["game::PlayerEntity"].base_classes == ("Entity",)
    assert len(elf_by_name["game::PlayerEntity"].method_rvas) == 3
    # Itanium vptr points to _ZTV + 16
    ztv_player_sym = next(
        s for s in suite.elf_binary.symbols if s.name == "_ZTVN4game12PlayerEntityE"
    )
    assert elf_by_name["game::PlayerEntity"].vtable_rva == ztv_player_sym.rva + 16


def test_binary_loader_rejects_corrupt_inputs() -> None:
    with pytest.raises(BinaryFormatError):
        load_binary(b"NOT_A_BINARY")
    with pytest.raises(BinaryFormatError):
        load_binary(b"MZ" + b"\x00" * 100)
    with pytest.raises(BinaryFormatError):
        load_binary(b"\x7fELF" + b"\x00" * 80)


def test_sfdump_roundtrip_and_corruption() -> None:
    suite = make_demo_suite()
    snap = suite.snapshots[0]
    raw = serialize_snapshot(snap)
    parsed = parse_snapshot(raw)

    assert parsed.image_base == snap.image_base
    assert parsed.root_va == snap.root_va
    assert len(parsed.regions) == len(snap.regions)
    assert parsed.read_va(snap.root_va, 8) == snap.read_va(snap.root_va, 8)

    with pytest.raises(SnapshotFormatError):
        parse_snapshot(b"BAD!")
    with pytest.raises(SnapshotFormatError):
        parse_snapshot(b"SFMD" + struct.pack("<IIQQI", 99, 0, 0, 0, 0))
    with pytest.raises(SnapshotFormatError):
        parse_snapshot(raw[:-10])
    with pytest.raises(SnapshotFormatError):
        parse_snapshot(raw + b"\x00")


def test_relocate_clone_preserves_pointers_and_vptrs() -> None:
    suite = make_demo_suite()
    snap = suite.snapshots[0]
    vtable_rvas = set(suite.vtable_rvas.values())
    img_delta = 0x100000
    heap_delta = 0x50000

    reloc = snap.relocate_clone(img_delta, heap_delta, vtable_rvas)
    assert reloc.image_base == snap.image_base + img_delta
    assert reloc.root_va == snap.root_va + img_delta

    world_va = struct.unpack("<Q", reloc.read_va(reloc.root_va, 8))[0]
    assert world_va == suite.world_vas[0] + heap_delta
    session_va = struct.unpack("<Q", reloc.read_va(world_va + 0x18, 8))[0]
    assert session_va == suite.session_vas[0] + heap_delta
    player_va = struct.unpack("<Q", reloc.read_va(session_va + 0x40, 8))[0]
    assert player_va == suite.player_vas[0] + heap_delta

    player_vptr = struct.unpack("<Q", reloc.read_va(player_va, 8))[0]
    assert player_vptr - reloc.image_base == suite.vtable_rvas["game::PlayerEntity"]
    cstr_va = struct.unpack("<Q", reloc.read_va(player_va + 0x20, 8))[0]
    assert reloc.read_cstr_va(cstr_va) == "ACTUAL_HERO_01"


def test_validate_spec_rejects_keywords_and_bad_sizes() -> None:
    with pytest.raises(SpecValidationError):
        SdkSpec.from_dict(
            {
                "name": "class",
                "structs": [
                    {
                        "name": "Player",
                        "size": 4,
                        "alignment": 4,
                        "fields": [{"name": "hp", "offset": 0, "size": 4, "kind": "f32"}],
                    }
                ],
            }
        )

    with pytest.raises(SpecValidationError):
        SdkSpec.from_dict(
            {
                "name": "ValidSdk",
                "structs": [
                    {
                        "name": "Player",
                        "size": 8,
                        "alignment": 3,
                        "fields": [{"name": "hp", "offset": 0, "size": 4, "kind": "f32"}],
                    }
                ],
            }
        )

    with pytest.raises(SpecValidationError):
        SdkSpec.from_dict(
            {
                "name": "ValidSdk",
                "structs": [
                    {
                        "name": "Player",
                        "size": 8,
                        "alignment": 4,
                        "fields": [{"name": "hp", "offset": 0, "size": 8, "kind": "f32"}],
                    }
                ],
            }
        )

    valid = SdkSpec(
        name="ValidSdk",
        structs=[
            StructDef(
                name="Player",
                size=4,
                alignment=4,
                fields=[StructField(name="hp", offset=0, size=4, kind=FieldKind.F32)],
            )
        ],
    )
    assert SdkSpec.from_json(valid.to_json()).name == "ValidSdk"
    # Ensure HeapSnapshot read_cstr_va requires len >= 3
    s = HeapSnapshot(
        image_base=0,
        root_va=0x1000,
        regions=[MemoryRegion(name="r", base_va=0x1000, data=b"AB\x00XYZ\x00")],
    )
    assert s.read_cstr_va(0x1000) is None
    assert s.read_cstr_va(0x1003) == "XYZ"

    # PointerChain cannot have variable-length FieldKind (CHAR_ARRAY / PAD)
    with pytest.raises(SpecValidationError):
        SdkSpec.from_dict(
            {
                "name": "BadChainKind",
                "structs": [
                    {
                        "name": "A",
                        "size": 16,
                        "alignment": 1,
                        "fields": [{"name": "txt", "offset": 0, "size": 16, "kind": "char_array"}],
                    }
                ],
                "pointer_chains": [
                    {
                        "name": "txt",
                        "root_rva": 0x1000,
                        "offsets": [0x0],
                        "field_kind": "char_array",
                    }
                ],
            }
        )

    # relocate_clone leaves flags=0 root_va unshifted
    s_static = HeapSnapshot(
        image_base=0x1000,
        root_va=0x9000,
        regions=[MemoryRegion(name="fixed", base_va=0x9000, data=b"\x00" * 16, flags=0)],
    )
    assert s_static.relocate_clone(0x10000, 0x20000, set()).root_va == 0x9000
