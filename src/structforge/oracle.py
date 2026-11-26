from __future__ import annotations

import math
import random
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from structforge.binary_loader import LoadedBinary
from structforge.models import FIELD_ALIGNMENT, FIXED_FIELD_SIZES, FieldKind, PointerChain, SdkSpec
from structforge.rtti import harvest_rtti
from structforge.snapshot import HeapSnapshot


@dataclass
class OracleResult:
    passed: bool
    gate1_abi_layout: bool
    gate2_snapshot_replay: bool
    gate3_aslr_and_drift: bool
    aslr_pass_rate: float
    drift_reanchor_rate: float
    layout_score: float
    failures: list[str] = field(default_factory=list)


def _read_chain_value(snap: HeapSnapshot, ch: PointerChain) -> Any | None:
    base_raw = snap.read_va(snap.image_base + ch.root_rva, 8)
    if base_raw is None:
        return None
    ptr = struct.unpack("<Q", base_raw)[0]
    for off in ch.offsets[:-1]:
        next_raw = snap.read_va(ptr + off, 8)
        if next_raw is None:
            return None
        ptr = struct.unpack("<Q", next_raw)[0]

    leaf_va = ptr + ch.offsets[-1]
    fsize = FIXED_FIELD_SIZES.get(ch.field_kind)
    if fsize is None:
        return None
    leaf_raw = snap.read_va(leaf_va, fsize)
    if leaf_raw is None:
        return None

    if ch.field_kind == FieldKind.F32:
        return struct.unpack("<f", leaf_raw)[0]
    if ch.field_kind == FieldKind.F64:
        return struct.unpack("<d", leaf_raw)[0]
    if ch.field_kind == FieldKind.BOOL:
        if leaf_raw[0] not in (0, 1):
            return None
        return bool(leaf_raw[0])
    if ch.field_kind == FieldKind.I32:
        return struct.unpack("<i", leaf_raw)[0]
    if ch.field_kind == FieldKind.U32:
        return struct.unpack("<I", leaf_raw)[0]
    if ch.field_kind == FieldKind.I64:
        return struct.unpack("<q", leaf_raw)[0]
    if ch.field_kind in (FieldKind.U64, FieldKind.PTR64, FieldKind.VPTR):
        return struct.unpack("<Q", leaf_raw)[0]
    if ch.field_kind == FieldKind.U16:
        return struct.unpack("<H", leaf_raw)[0]
    if ch.field_kind == FieldKind.U8:
        return leaf_raw[0]
    if ch.field_kind == FieldKind.CSTR_PTR:
        return snap.read_cstr_va(struct.unpack("<Q", leaf_raw)[0])
    if ch.field_kind == FieldKind.STD_VECTOR:
        b, e, c = struct.unpack("<QQQ", leaf_raw)
        if not (b != 0 and b <= e <= c and snap.contains_va(b)):
            return None
        return e - b
    return None


def _values_equal(kind: FieldKind, actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    if kind in (FieldKind.F32, FieldKind.F64):
        return math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= 1e-4
    if kind == FieldKind.BOOL:
        return bool(actual) == bool(expected)
    if kind == FieldKind.CSTR_PTR:
        return str(actual) == str(expected)
    return int(actual) == int(expected)


def _walk_leaf_va(snap: HeapSnapshot, ch: PointerChain) -> int | None:
    base_raw = snap.read_va(snap.image_base + ch.root_rva, 8)
    if base_raw is None:
        return None
    ptr = struct.unpack("<Q", base_raw)[0]
    for off in ch.offsets[:-1]:
        next_raw = snap.read_va(ptr + off, 8)
        if next_raw is None:
            return None
        ptr = struct.unpack("<Q", next_raw)[0]
    return ptr


def _read_from_buf(
    buf: bytes, off: int, kind: FieldKind, snap: HeapSnapshot | None = None
) -> Any | None:
    fsize = FIXED_FIELD_SIZES.get(kind)
    if fsize is None or off < 0 or off + fsize > len(buf):
        return None
    raw = buf[off : off + fsize]
    if kind == FieldKind.F32:
        return struct.unpack("<f", raw)[0]
    if kind == FieldKind.F64:
        return struct.unpack("<d", raw)[0]
    if kind == FieldKind.BOOL:
        if raw[0] not in (0, 1):
            return None
        return bool(raw[0])
    if kind == FieldKind.I32:
        return struct.unpack("<i", raw)[0]
    if kind == FieldKind.U32:
        return struct.unpack("<I", raw)[0]
    if kind == FieldKind.I64:
        return struct.unpack("<q", raw)[0]
    if kind in (FieldKind.U64, FieldKind.PTR64, FieldKind.VPTR):
        return struct.unpack("<Q", raw)[0]
    if kind == FieldKind.U16:
        return struct.unpack("<H", raw)[0]
    if kind == FieldKind.U8:
        return raw[0]
    if kind == FieldKind.CSTR_PTR:
        if snap is None:
            return None
        return snap.read_cstr_va(struct.unpack("<Q", raw)[0])
    if kind == FieldKind.STD_VECTOR:
        b, e, c = struct.unpack("<QQQ", raw)
        if not (b != 0 and b <= e <= c):
            return None
        if snap is not None and not snap.contains_va(b):
            return None
        return e - b
    return None


def verify_sdk(
    spec: SdkSpec,
    snapshots: Sequence[HeapSnapshot],
    binary: LoadedBinary | None = None,
    *,
    trials: int = 100,
    seed: int = 0x5746049E,
) -> OracleResult:
    failures: list[str] = []

    # Gate 1: Strict x86-64 ABI Layout & Anti-Triviality
    gate1 = True
    if not spec.structs:
        gate1 = False
        failures.append("Gate 1: SdkSpec contains no structs")

    densities: list[float] = []
    for st in spec.structs:
        if st.size <= 0 or st.alignment not in (1, 2, 4, 8):
            gate1 = False
            failures.append(
                f"Gate 1: Struct '{st.name}' has invalid size ({st.size}) or alignment ({st.alignment})"
            )
            continue
        cur = 0
        for fld in st.fields:
            if fld.offset != cur:
                gate1 = False
                failures.append(
                    f"Gate 1: Gap or overlap in '{st.name}.{fld.name}': expected offset 0x{cur:x}, got 0x{fld.offset:x}"
                )
                cur = fld.offset
            if fld.offset % fld.alignment != 0:
                gate1 = False
                failures.append(
                    f"Gate 1: Misaligned field '{st.name}.{fld.name}' at 0x{fld.offset:x} (align {fld.alignment})"
                )
            cur += fld.size
        if cur != st.size:
            gate1 = False
            failures.append(
                f"Gate 1: Struct '{st.name}' field span 0x{cur:x} != declared size 0x{st.size:x}"
            )
        expected_align = max((f.alignment for f in st.fields), default=1)
        if st.alignment != expected_align:
            gate1 = False
            failures.append(
                f"Gate 1: Struct '{st.name}' alignment {st.alignment} != max field alignment {expected_align}"
            )
        if st.size % st.alignment != 0:
            gate1 = False
            failures.append(
                f"Gate 1: Struct '{st.name}' size 0x{st.size:x} is not a multiple of alignment {st.alignment}"
            )
        non_pad = sum(f.size for f in st.fields if f.kind != FieldKind.PAD)
        if non_pad <= 0 or (non_pad / st.size) < 0.15:
            gate1 = False
            failures.append(
                f"Gate 1: Struct '{st.name}' semantic density {non_pad}/{st.size} < 15%"
            )
        densities.append(non_pad / st.size)

    # Gate 2: Multi-Snapshot Replay & RTTI VPtr Proof
    gate2 = True
    if not snapshots:
        gate2 = False
        failures.append("Gate 2: No snapshots provided for replay")
    else:
        for ch in spec.pointer_chains:
            for s_idx, snap in enumerate(snapshots):
                actual = _read_chain_value(snap, ch)
                if actual is None:
                    gate2 = False
                    failures.append(
                        f"Gate 2: PointerChain '{ch.name}' failed to dereference in snapshot {s_idx}"
                    )
                elif s_idx < len(ch.expected_values) and not _values_equal(
                    ch.field_kind, actual, ch.expected_values[s_idx]
                ):
                    gate2 = False
                    failures.append(
                        f"Gate 2: PointerChain '{ch.name}' mismatch in snapshot {s_idx}: got {actual!r}, expected {ch.expected_values[s_idx]!r}"
                    )

        for st in spec.structs:
            if not st.instance_vas:
                continue
            if len(st.instance_vas) != len(snapshots):
                gate2 = False
                failures.append(
                    f"Gate 2: Struct '{st.name}' instance_vas count {len(st.instance_vas)} != snapshots {len(snapshots)}"
                )
                continue
            for s_idx, (snap, inst_va) in enumerate(zip(snapshots, st.instance_vas)):
                for fld in st.fields:
                    if fld.kind == FieldKind.VPTR and fld.vtable_rva is not None:
                        raw = snap.read_va(inst_va + fld.offset, 8)
                        expected_vptr = snap.image_base + fld.vtable_rva
                        if raw is None or struct.unpack("<Q", raw)[0] != expected_vptr:
                            gate2 = False
                            failures.append(
                                f"Gate 2: VPtr mismatch on '{st.name}.{fld.name}' in snapshot {s_idx}"
                            )
                    elif fld.kind == FieldKind.STD_VECTOR:
                        raw = snap.read_va(inst_va + fld.offset, 24)
                        if raw is None:
                            gate2 = False
                            failures.append(
                                f"Gate 2: Unable to read std_vector '{st.name}.{fld.name}' in snapshot {s_idx}"
                            )
                        else:
                            b, e, c = struct.unpack("<QQQ", raw)
                            if not (b != 0 and b <= e <= c):
                                gate2 = False
                                failures.append(
                                    f"Gate 2: Invalid std_vector bounds on '{st.name}.{fld.name}' in snapshot {s_idx}"
                                )

    # Gate 3: 100-Trial Deterministic ASLR Relocation & Struct-Drift Fuzzer
    vtable_rvas = {v.vtable_rva for v in spec.vtables} | {
        f.vtable_rva for s in spec.structs for f in s.fields if f.vtable_rva is not None
    }
    if binary is not None:
        vtable_rvas |= {vt.vtable_rva for vt in harvest_rtti(binary)}

    rng = random.Random(seed)
    aslr_pass = 0
    safe_trials = max(1, trials)

    for _ in range(safe_trials):
        img_delta = rng.randint(1, 64) * 0x10000
        heap_delta = rng.randint(1, 256) * 0x1000
        trial_ok = bool(snapshots)
        if trial_ok:
            reloc_snaps = [
                snap.relocate_clone(img_delta, heap_delta, vtable_rvas) for snap in snapshots
            ]
            for ch in spec.pointer_chains:
                for s_idx, rsnap in enumerate(reloc_snaps):
                    val = _read_chain_value(rsnap, ch)
                    if val is None or (
                        s_idx < len(ch.expected_values)
                        and not _values_equal(ch.field_kind, val, ch.expected_values[s_idx])
                    ):
                        trial_ok = False
                        break
                if not trial_ok:
                    break

            if trial_ok:
                for st in spec.structs:
                    if len(st.instance_vas) != len(snapshots):
                        continue
                    for snap, rsnap, inst_va in zip(snapshots, reloc_snaps, st.instance_vas):
                        reg = snap.region_at_va(inst_va)
                        delta = img_delta if (reg is not None and (reg.flags & 2)) else heap_delta
                        reloc_va = inst_va + delta
                        for fld in st.fields:
                            if fld.kind == FieldKind.VPTR and fld.vtable_rva is not None:
                                raw = rsnap.read_va(reloc_va + fld.offset, 8)
                                if (
                                    raw is None
                                    or (struct.unpack("<Q", raw)[0] - rsnap.image_base)
                                    != fld.vtable_rva
                                ):
                                    trial_ok = False
                                    break
                        if not trial_ok:
                            break
                    if not trial_ok:
                        break
        if trial_ok:
            aslr_pass += 1

    aslr_pass_rate = aslr_pass / safe_trials

    # Patch Struct-Drift Re-Anchoring
    var_chains = [
        ch
        for ch in spec.pointer_chains
        if len(ch.expected_values) >= len(snapshots)
        and len({str(v) for v in ch.expected_values}) > 1
    ]
    drift_trials = max(1, safe_trials // 2)
    if not var_chains or len(snapshots) < 2:
        drift_reanchor_rate = 1.0
    else:
        drift_pass = 0
        for _ in range(drift_trials):
            insert_sz = rng.choice((8, 16))
            all_chains_reanchored = True
            for ch in var_chains:
                orig_leaf_off = ch.offsets[-1]
                fsize = FIXED_FIELD_SIZES.get(ch.field_kind, 8)
                slice_len = max(0x100, orig_leaf_off + fsize + 0x20)
                leaf_bufs: list[bytes] = []
                for snap in snapshots:
                    leaf_va = _walk_leaf_va(snap, ch)
                    if leaf_va is None:
                        all_chains_reanchored = False
                        break
                    reg = snap.region_at_va(leaf_va)
                    if reg is None:
                        all_chains_reanchored = False
                        break
                    off_in_reg = leaf_va - reg.base_va
                    chunk = reg.data[off_in_reg : off_in_reg + slice_len]
                    drifted = chunk[:0x08] + (b"\x00" * insert_sz) + chunk[0x08:]
                    leaf_bufs.append(drifted)
                if not all_chains_reanchored:
                    break

                expected_new_off = (
                    orig_leaf_off + insert_sz if orig_leaf_off >= 0x08 else orig_leaf_off
                )
                align = FIELD_ALIGNMENT[ch.field_kind]
                matched_offsets: list[int] = []
                min_len = min(len(b) for b in leaf_bufs)
                for cand_off in range(0, min_len, align):
                    if all(
                        _values_equal(
                            ch.field_kind,
                            _read_from_buf(buf, cand_off, ch.field_kind, snap),
                            exp,
                        )
                        for buf, snap, exp in zip(leaf_bufs, snapshots, ch.expected_values)
                    ):
                        matched_offsets.append(cand_off)
                if expected_new_off not in matched_offsets:
                    all_chains_reanchored = False
                    break
            if all_chains_reanchored:
                drift_pass += 1
        drift_reanchor_rate = drift_pass / drift_trials

    semantic_density = sum(densities) / len(densities) if densities else 0.0
    layout_score = round(
        min(
            100.0,
            50.0 * aslr_pass_rate
            + 25.0 * min(1.0, semantic_density / 0.50)
            + 25.0 * drift_reanchor_rate,
        ),
        2,
    )

    gate3 = (aslr_pass_rate == 1.0) and (layout_score >= 80.0)
    if aslr_pass_rate < 1.0:
        failures.append(f"Gate 3: ASLR relocation pass rate {aslr_pass_rate:.2%} < 100%")
    if layout_score < 80.0:
        failures.append(f"Gate 3: Layout score {layout_score} < 80.0")

    passed = gate1 and gate2 and gate3
    return OracleResult(
        passed=passed,
        gate1_abi_layout=gate1,
        gate2_snapshot_replay=gate2,
        gate3_aslr_and_drift=gate3,
        aslr_pass_rate=aslr_pass_rate,
        drift_reanchor_rate=drift_reanchor_rate,
        layout_score=layout_score,
        failures=failures,
    )
