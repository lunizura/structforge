from __future__ import annotations

import struct

import pytest

from structforge.errors import InferenceError
from structforge.infer import WatchSpec, find_pointer_chains, infer_struct, synthesize_sdk
from structforge.models import FieldKind, PointerChain, SdkSpec, StructDef, StructField
from structforge.oracle import verify_sdk
from structforge.rtti import harvest_rtti
from structforge.snapshot import HeapSnapshot, MemoryRegion
from structforge.synthetic_fixtures import make_demo_suite


def test_infer_player_entity_and_tail_alignment() -> None:
    suite = make_demo_suite()
    vts = harvest_rtti(suite.pe_binary)

    st = infer_struct(
        name="PlayerEntity",
        obj_vas=suite.player_vas,
        obj_size=0x48,
        snapshots=suite.snapshots,
        vtables=vts,
        field_overrides={
            0x10: ("health", FieldKind.F32),
            0x18: ("ammo", FieldKind.I32),
            0x1C: ("is_alive", FieldKind.BOOL),
        },
    )
    assert st.size == 0x48
    assert st.alignment == 8
    assert st.demangled_rtti == "game::PlayerEntity"
    assert st.base_classes == ["engine::Entity"]

    by_off = {f.offset: f for f in st.fields}
    assert by_off[0x00].kind == FieldKind.VPTR
    assert by_off[0x08].kind == FieldKind.I32
    assert by_off[0x0C].kind == FieldKind.I32
    assert by_off[0x10].kind == FieldKind.F32
    assert by_off[0x10].name == "health"
    assert by_off[0x14].kind == FieldKind.F32
    assert by_off[0x18].kind == FieldKind.I32
    assert by_off[0x18].name == "ammo"
    assert by_off[0x1C].kind == FieldKind.BOOL
    assert by_off[0x1C].name == "is_alive"
    assert by_off[0x1D].kind == FieldKind.PAD
    assert by_off[0x1D].size == 3
    assert by_off[0x20].kind == FieldKind.CSTR_PTR
    assert by_off[0x20].sample_values == ["ACTUAL_HERO_01"] * 3
    assert by_off[0x28].kind == FieldKind.STD_VECTOR
    assert by_off[0x28].elem_stride == 4
    assert by_off[0x40].kind == FieldKind.F64

    # Tail-alignment invariant on unaligned obj_size
    unaligned = infer_struct(
        name="PlayerUnaligned",
        obj_vas=suite.player_vas,
        obj_size=0x43,
        snapshots=suite.snapshots,
        vtables=vts,
    )
    assert unaligned.alignment == 8
    assert unaligned.size == 0x48
    assert unaligned.size % unaligned.alignment == 0


def test_bfs_pointer_chain_discovery_and_synthesize_sdk() -> None:
    suite = make_demo_suite()
    watches = [
        WatchSpec(name="health", kind=FieldKind.F32, values=(100.0, 72.5, 0.0)),
        WatchSpec(name="ammo", kind=FieldKind.I32, values=(30, 29, 0)),
        WatchSpec(name="is_alive", kind=FieldKind.BOOL, values=(True, True, False)),
    ]

    chains = find_pointer_chains(watches, suite.snapshots)
    by_name = {c.name: c for c in chains}
    assert by_name["health"].offsets == [0x18, 0x40, 0x10]
    assert by_name["ammo"].offsets == [0x18, 0x40, 0x18]
    assert by_name["is_alive"].offsets == [0x18, 0x40, 0x1C]

    spec, oracle = synthesize_sdk(
        snapshots=suite.snapshots,
        binary=suite.pe_binary,
        watches=watches,
        name="GameSdk",
        module_name="game_rtti.dll",
    )
    assert oracle.passed is True
    assert oracle.gate1_abi_layout is True
    assert oracle.gate2_snapshot_replay is True
    assert oracle.gate3_aslr_and_drift is True
    assert oracle.aslr_pass_rate == 1.0
    assert oracle.drift_reanchor_rate == 1.0
    assert oracle.layout_score >= 85.0

    struct_names = {s.name for s in spec.structs}
    assert {"WorldContext", "Session", "PlayerEntity"}.issubset(struct_names)
    player = next(s for s in spec.structs if s.name == "PlayerEntity")
    player_fields = {f.name: f for f in player.fields}
    assert player_fields["health"].offset == 0x10
    assert player_fields["ammo"].offset == 0x18
    assert player_fields["is_alive"].offset == 0x1C


def test_oracle_rejects_bad_abi_and_bad_replay() -> None:
    suite = make_demo_suite()

    # Gate 1 failure: misaligned f32 at offset 2 and overlapping/gap
    bad_abi_spec = SdkSpec(
        name="BadAbi",
        structs=[
            StructDef(
                name="Broken",
                size=8,
                alignment=4,
                fields=[
                    StructField(name="a", offset=0, size=2, kind=FieldKind.U16),
                    StructField(name="b", offset=2, size=4, kind=FieldKind.F32),
                    StructField(name="c", offset=6, size=2, kind=FieldKind.U16),
                ],
            )
        ],
    )
    res_abi = verify_sdk(bad_abi_spec, suite.snapshots, trials=10)
    assert res_abi.gate1_abi_layout is False
    assert res_abi.passed is False

    # Gate 1 failure: non-multiple tail size
    bad_tail_spec = SdkSpec(
        name="BadTail",
        structs=[
            StructDef(
                name="Unpadded",
                size=5,
                alignment=4,
                fields=[
                    StructField(name="hp", offset=0, size=4, kind=FieldKind.F32),
                    StructField(name="flag", offset=4, size=1, kind=FieldKind.BOOL),
                ],
            )
        ],
    )
    res_tail = verify_sdk(bad_tail_spec, suite.snapshots, trials=10)
    assert res_tail.gate1_abi_layout is False
    assert res_tail.passed is False

    # Gate 2 failure: wrong pointer chain expected value
    bad_chain_spec = SdkSpec(
        name="BadChain",
        structs=[
            StructDef(
                name="OkStruct",
                size=8,
                alignment=4,
                fields=[
                    StructField(name="a", offset=0, size=4, kind=FieldKind.I32),
                    StructField(name="b", offset=4, size=4, kind=FieldKind.F32),
                ],
            )
        ],
        pointer_chains=[
            PointerChain(
                name="health",
                root_rva=0x5000,
                offsets=[0x18, 0x40, 0x10],
                field_kind=FieldKind.F32,
                expected_values=[999.0, 999.0, 999.0],
            )
        ],
    )
    res_chain = verify_sdk(bad_chain_spec, suite.snapshots, trials=10)
    assert res_chain.gate1_abi_layout is True
    assert res_chain.gate2_snapshot_replay is False
    assert res_chain.passed is False

    # Gate 1 failure: zero-size empty struct must not raise ZeroDivisionError
    empty_spec = SdkSpec(
        name="EmptySpec",
        structs=[StructDef(name="Empty", size=0, alignment=1, fields=[])],
    )
    res_empty = verify_sdk(empty_spec, suite.snapshots, trials=10)
    assert res_empty.gate1_abi_layout is False
    assert res_empty.passed is False


def test_override_overlap_and_gate3_cstr_and_high_offset() -> None:
    # 1. field_overrides at +0x04 must not be swallowed by F64 at +0x00
    buf0 = bytearray(0x10)
    buf1 = bytearray(0x10)
    struct.pack_into("<d", buf0, 0x00, 142.5000001)
    struct.pack_into("<d", buf1, 0x00, 143.2500001)
    s0 = HeapSnapshot(
        image_base=0, root_va=0x1000, regions=[MemoryRegion("h", 0x1000, bytes(buf0))]
    )
    s1 = HeapSnapshot(
        image_base=0, root_va=0x1000, regions=[MemoryRegion("h", 0x1000, bytes(buf1))]
    )
    st = infer_struct(
        "SplitF64",
        [0x1000, 0x1000],
        0x10,
        [s0, s1],
        field_overrides={0x04: ("hi_dword", FieldKind.U32)},
    )
    by_off = {f.offset: f for f in st.fields}
    assert by_off[0x04].name == "hi_dword"
    assert by_off[0x04].kind == FieldKind.U32

    # 2. Gate 3 drift re-anchoring supports CSTR_PTR and leaf offsets >= 0x100
    snaps: list[HeapSnapshot] = []
    for idx, (label, hp) in enumerate(
        [("HERO_ALPHA", 100.0), ("HERO_BRAVO", 72.5), ("HERO_CHARLIE", 0.0)]
    ):
        base = 0x20000000 + idx * 0x10000
        h = bytearray(0x300)
        struct.pack_into("<Q", h, 0x08, base + 0x240)
        struct.pack_into("<f", h, 0x140, hp)
        raw_lbl = label.encode("ascii") + b"\x00"
        h[0x240 : 0x240 + len(raw_lbl)] = raw_lbl
        d = bytearray(0x20)
        struct.pack_into("<Q", d, 0, base)
        snaps.append(
            HeapSnapshot(
                image_base=0x140000000,
                root_va=0x140005000,
                regions=[
                    MemoryRegion(".data", 0x140005000, bytes(d), flags=2),
                    MemoryRegion("heap", base, bytes(h), flags=1),
                ],
            )
        )

    spec_cstr, res_cstr = synthesize_sdk(
        snaps,
        watches=[
            WatchSpec(
                "hero_name", FieldKind.CSTR_PTR, ("HERO_ALPHA", "HERO_BRAVO", "HERO_CHARLIE")
            ),
            WatchSpec("health", FieldKind.F32, (100.0, 72.5, 0.0)),
        ],
    )
    assert res_cstr.drift_reanchor_rate == 1.0
    assert res_cstr.aslr_pass_rate == 1.0
    assert spec_cstr.pointer_chains[1].offsets[-1] == 0x140

    # 3. Unmatched watch must raise InferenceError (never silently dropped)
    with pytest.raises(InferenceError, match="nonexistent_stat"):
        synthesize_sdk(
            snaps,
            watches=[
                WatchSpec("health", FieldKind.F32, (100.0, 72.5, 0.0)),
                WatchSpec("nonexistent_stat", FieldKind.I32, (999, 888, 777)),
            ],
        )

    # 4. Adjacent heap string allocation must not bleed into parent struct size
    adj_snaps: list[HeapSnapshot] = []
    for hp in (100.0, 72.5, 0.0):
        hbase = 0x20000000
        h = bytearray(0x100)
        struct.pack_into("<i", h, 0x00, 1001)
        struct.pack_into("<f", h, 0x10, hp)
        struct.pack_into("<Q", h, 0x20, hbase + 0x30)
        h[0x30:0x3F] = b"SEPARATE_ALLOC\x00"
        d = bytearray(0x10)
        struct.pack_into("<Q", d, 0, hbase)
        adj_snaps.append(
            HeapSnapshot(
                0x140000000,
                0x140005000,
                [
                    MemoryRegion(".data", 0x140005000, bytes(d), 2),
                    MemoryRegion("heap", hbase, bytes(h), 1),
                ],
            )
        )
    sp_adj, res_adj = synthesize_sdk(
        adj_snaps, watches=[WatchSpec("hp", FieldKind.F32, (100.0, 72.5, 0.0))]
    )
    assert res_adj.passed is True
    assert sp_adj.structs[0].size == 0x28
    assert all(f.offset < 0x28 for f in sp_adj.structs[0].fields)
