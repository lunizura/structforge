from __future__ import annotations

import math
import struct
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from structforge.binary_loader import LoadedBinary
from structforge.errors import InferenceError
from structforge.models import (
    FIELD_ALIGNMENT,
    FIXED_FIELD_SIZES,
    FieldKind,
    PointerChain,
    SdkSpec,
    StructDef,
    StructField,
    VTableInfo,
    sanitize_ident,
    validate_spec,
)
from structforge.oracle import OracleResult, verify_sdk
from structforge.rtti import harvest_rtti
from structforge.snapshot import HeapSnapshot


@dataclass(frozen=True)
class WatchSpec:
    name: str
    kind: FieldKind
    values: tuple[Any, ...]


def _is_plausible_f32_bits(u32: int) -> bool:
    exp = (u32 >> 23) & 0xFF
    return 100 <= exp <= 155


def _is_plausible_f64_bits(u64: int) -> bool:
    exp = (u64 >> 52) & 0x7FF
    lo32 = u64 & 0xFFFFFFFF
    hi32 = (u64 >> 32) & 0xFFFFFFFF
    return (
        1010 <= exp <= 1050
        and lo32 != 0
        and not (_is_plausible_f32_bits(lo32) and _is_plausible_f32_bits(hi32))
    )


def _unpack_override_samples(
    slices: Sequence[bytes],
    off: int,
    kind: FieldKind,
    snapshots: Sequence[HeapSnapshot],
    vt_by_rva: dict[int, VTableInfo],
) -> tuple[list[Any], str | None, int | None, int | None]:
    target_type: str | None = None
    vtable_rva: int | None = None
    elem_stride: int | None = None
    samples: list[Any] = []

    if kind == FieldKind.VPTR:
        q_vals = [struct.unpack_from("<Q", sl, off)[0] for sl in slices]
        rva0 = q_vals[0] - snapshots[0].image_base
        if rva0 in vt_by_rva and all(q - s.image_base == rva0 for q, s in zip(q_vals, snapshots)):
            vtable_rva = rva0
            target_type = vt_by_rva[rva0].demangled_name
        samples = [hex(q) for q in q_vals]
    elif kind == FieldKind.STD_VECTOR:
        triples = [struct.unpack_from("<QQQ", sl, off) for sl in slices]
        lens = [e - b for b, e, _ in triples if e > b]
        elem_stride = math.gcd(*lens) if lens else 1
        samples = [e - b for b, e, _ in triples]
    elif kind == FieldKind.CSTR_PTR:
        q_vals = [struct.unpack_from("<Q", sl, off)[0] for sl in slices]
        samples = [s.read_cstr_va(q) or hex(q) for q, s in zip(q_vals, snapshots)]
    elif kind in (FieldKind.PTR64, FieldKind.U64):
        samples = [struct.unpack_from("<Q", sl, off)[0] for sl in slices]
        if kind == FieldKind.PTR64:
            target_type = "void"
    elif kind == FieldKind.I64:
        samples = [struct.unpack_from("<q", sl, off)[0] for sl in slices]
    elif kind == FieldKind.F64:
        samples = [round(struct.unpack_from("<d", sl, off)[0], 6) for sl in slices]
    elif kind == FieldKind.F32:
        samples = [round(struct.unpack_from("<f", sl, off)[0], 4) for sl in slices]
    elif kind == FieldKind.I32:
        samples = [struct.unpack_from("<i", sl, off)[0] for sl in slices]
    elif kind == FieldKind.U32:
        samples = [struct.unpack_from("<I", sl, off)[0] for sl in slices]
    elif kind == FieldKind.U16:
        samples = [struct.unpack_from("<H", sl, off)[0] for sl in slices]
    elif kind == FieldKind.U8:
        samples = [sl[off] for sl in slices]
    elif kind == FieldKind.BOOL:
        samples = [bool(sl[off]) for sl in slices]

    return samples, target_type, vtable_rva, elem_stride


def infer_struct(
    name: str,
    obj_vas: Sequence[int],
    obj_size: int,
    snapshots: Sequence[HeapSnapshot],
    vtables: Sequence[VTableInfo] = (),
    field_overrides: dict[int, tuple[str, FieldKind]] | None = None,
) -> StructDef:
    if len(snapshots) < 1:
        raise InferenceError("infer_struct requires at least one HeapSnapshot")
    if len(obj_vas) != len(snapshots):
        raise InferenceError("obj_vas length must match snapshots length")
    if obj_size <= 0:
        raise InferenceError(f"obj_size must be positive, got {obj_size}")

    slices: list[bytes] = []
    for snap, va in zip(snapshots, obj_vas):
        raw = snap.read_va(va, obj_size)
        if raw is None:
            raise InferenceError(f"Unable to read {obj_size} bytes at 0x{va:x} in snapshot")
        slices.append(raw)

    vt_by_rva = {vt.vtable_rva: vt for vt in vtables}
    overrides = field_overrides or {}

    fields: list[StructField] = []
    unclaimed_start: int | None = None

    def _flush_pad(end_off: int) -> None:
        nonlocal unclaimed_start
        if unclaimed_start is not None and end_off > unclaimed_start:
            pad_size = end_off - unclaimed_start
            fields.append(
                StructField(
                    name=f"_pad_{unclaimed_start:04x}",
                    offset=unclaimed_start,
                    size=pad_size,
                    kind=FieldKind.PAD,
                    confidence=1.0,
                )
            )
        unclaimed_start = None

    def _overlaps_override(start: int, span: int) -> bool:
        return any(o in overrides for o in range(start, start + span))

    off = 0
    while off < obj_size:
        # Priority 0: Explicit field_overrides
        if off in overrides:
            ovr_name, ovr_kind = overrides[off]
            fsize = FIXED_FIELD_SIZES.get(ovr_kind)
            if (
                fsize is not None
                and off + fsize <= obj_size
                and off % FIELD_ALIGNMENT[ovr_kind] == 0
            ):
                _flush_pad(off)
                samples, ttype, vrva, stride = _unpack_override_samples(
                    slices, off, ovr_kind, snapshots, vt_by_rva
                )
                fields.append(
                    StructField(
                        name=sanitize_ident(ovr_name),
                        offset=off,
                        size=fsize,
                        kind=ovr_kind,
                        target_type=ttype,
                        vtable_rva=vrva,
                        elem_stride=stride,
                        sample_values=samples,
                        confidence=1.0,
                    )
                )
                off += fsize
                continue

        # Priority 1: VPTR (off % 8 == 0)
        if off % 8 == 0 and off + 8 <= obj_size and not _overlaps_override(off + 1, 7):
            q_vals = [struct.unpack_from("<Q", sl, off)[0] for sl in slices]
            rva0 = q_vals[0] - snapshots[0].image_base
            if rva0 in vt_by_rva and all(
                q - snap.image_base == rva0 for q, snap in zip(q_vals, snapshots)
            ):
                _flush_pad(off)
                vt = vt_by_rva[rva0]
                fields.append(
                    StructField(
                        name="vptr" if off == 0 else f"vptr_{off:04x}",
                        offset=off,
                        size=8,
                        kind=FieldKind.VPTR,
                        target_type=vt.demangled_name,
                        vtable_rva=rva0,
                        sample_values=[hex(q) for q in q_vals],
                        confidence=1.0,
                    )
                )
                off += 8
                continue

        # Priority 2: STD_VECTOR (off % 8 == 0, 24B)
        if off % 8 == 0 and off + 24 <= obj_size and not _overlaps_override(off + 1, 23):
            triples = [struct.unpack_from("<QQQ", sl, off) for sl in slices]
            if all(
                b != 0
                and b <= e <= c
                and 0 < (c - b) <= 0x100000
                and snap.contains_va(b)
                and (e == b or snap.contains_va(e - 1))
                for (b, e, c), snap in zip(triples, snapshots)
            ) and not all(
                snap.read_cstr_va(b) is not None for (b, _, _), snap in zip(triples, snapshots)
            ):
                _flush_pad(off)
                non_empty = [e - b for b, e, _ in triples if e > b]
                elem_stride = math.gcd(*non_empty) if non_empty else 1
                fields.append(
                    StructField(
                        name=f"vec_{off:04x}",
                        offset=off,
                        size=24,
                        kind=FieldKind.STD_VECTOR,
                        elem_stride=elem_stride,
                        sample_values=[e - b for b, e, _ in triples],
                        confidence=0.95,
                    )
                )
                off += 24
                continue

        # Priority 3: CSTR_PTR / PTR64 (off % 8 == 0)
        if off % 8 == 0 and off + 8 <= obj_size and not _overlaps_override(off + 1, 7):
            q_vals = [struct.unpack_from("<Q", sl, off)[0] for sl in slices]
            if all(q != 0 and snap.contains_va(q) for q, snap in zip(q_vals, snapshots)):
                _flush_pad(off)
                strs = [snap.read_cstr_va(q) for q, snap in zip(q_vals, snapshots)]
                if all(s is not None for s in strs):
                    fields.append(
                        StructField(
                            name=f"str_ptr_{off:04x}",
                            offset=off,
                            size=8,
                            kind=FieldKind.CSTR_PTR,
                            sample_values=[s for s in strs if s is not None],
                            confidence=0.95,
                        )
                    )
                    off += 8
                    continue

                target_type = "void"
                vptr_rvas: list[int | None] = []
                for q, snap in zip(q_vals, snapshots):
                    raw_vptr = snap.read_va(q, 8)
                    if raw_vptr is not None:
                        cand_rva = struct.unpack("<Q", raw_vptr)[0] - snap.image_base
                        vptr_rvas.append(cand_rva if cand_rva in vt_by_rva else None)
                    else:
                        vptr_rvas.append(None)
                if vptr_rvas[0] is not None and all(r == vptr_rvas[0] for r in vptr_rvas):
                    vt = vt_by_rva[vptr_rvas[0]]
                    target_type = sanitize_ident(vt.demangled_name.split("::")[-1])

                fields.append(
                    StructField(
                        name=f"ptr_{off:04x}",
                        offset=off,
                        size=8,
                        kind=FieldKind.PTR64,
                        target_type=target_type,
                        sample_values=[hex(q) for q in q_vals],
                        confidence=0.95,
                    )
                )
                off += 8
                continue

        # Priority 4: CHAR_ARRAY (off % 8 == 0 or off % 4 == 0)
        if off % 4 == 0:
            matched_str: tuple[int, list[str]] | None = None
            for buf_len in (16, 32, 64):
                if off + buf_len > obj_size or _overlaps_override(off + 1, buf_len - 1):
                    continue
                parsed_strs: list[str] = []
                ok = True
                for sl in slices:
                    chunk = sl[off : off + buf_len]
                    nul_idx = chunk.find(b"\x00")
                    if (
                        nul_idx < 4
                        or not all(0x20 <= b <= 0x7E for b in chunk[:nul_idx])
                        or chunk[nul_idx:] != b"\x00" * (buf_len - nul_idx)
                    ):
                        ok = False
                        break
                    parsed_strs.append(chunk[:nul_idx].decode("ascii"))
                if ok:
                    matched_str = (buf_len, parsed_strs)
                    break
            if matched_str is not None:
                _flush_pad(off)
                buf_len, parsed_strs = matched_str
                fields.append(
                    StructField(
                        name=f"text_{off:04x}",
                        offset=off,
                        size=buf_len,
                        kind=FieldKind.CHAR_ARRAY,
                        sample_values=parsed_strs,
                        confidence=0.9,
                    )
                )
                off += buf_len
                continue

        # Priority 5: F64 (off % 8 == 0)
        if off % 8 == 0 and off + 8 <= obj_size and not _overlaps_override(off + 1, 7):
            q_vals = [struct.unpack_from("<Q", sl, off)[0] for sl in slices]
            if any(q != 0 for q in q_vals) and all(
                q == 0 or _is_plausible_f64_bits(q) for q in q_vals
            ):
                _flush_pad(off)
                d_vals = [round(struct.unpack_from("<d", sl, off)[0], 6) for sl in slices]
                fields.append(
                    StructField(
                        name=f"f64_{off:04x}",
                        offset=off,
                        size=8,
                        kind=FieldKind.F64,
                        sample_values=d_vals,
                        confidence=0.9,
                    )
                )
                off += 8
                continue

        # Priority 6: BOOL / F32 / I32 / U32 (off % 4 == 0)
        if off % 4 == 0 and off + 4 <= obj_size:
            if all(
                sl[off] in (0, 1) and sl[off + 1 : off + 4] == b"\x00\x00\x00" for sl in slices
            ) and {sl[off] for sl in slices} == {0, 1}:
                _flush_pad(off)
                fields.append(
                    StructField(
                        name=f"flag_{off:04x}",
                        offset=off,
                        size=1,
                        kind=FieldKind.BOOL,
                        sample_values=[bool(sl[off]) for sl in slices],
                        confidence=0.9,
                    )
                )
                off += 1
                continue

            if not _overlaps_override(off + 1, 3):
                u_vals = [struct.unpack_from("<I", sl, off)[0] for sl in slices]
                if any(_is_plausible_f32_bits(u) for u in u_vals) and all(
                    u == 0 or _is_plausible_f32_bits(u) for u in u_vals
                ):
                    _flush_pad(off)
                    f_vals = [round(struct.unpack_from("<f", sl, off)[0], 4) for sl in slices]
                    fields.append(
                        StructField(
                            name=f"f32_{off:04x}",
                            offset=off,
                            size=4,
                            kind=FieldKind.F32,
                            sample_values=f_vals,
                            confidence=0.9,
                        )
                    )
                    off += 4
                    continue

                if any(u != 0 for u in u_vals):
                    _flush_pad(off)
                    i_vals = [struct.unpack_from("<i", sl, off)[0] for sl in slices]
                    if any(-100_000_000 <= iv < 0 for iv in i_vals) or all(
                        0 <= u <= 100_000_000 for u in u_vals
                    ):
                        fields.append(
                            StructField(
                                name=f"i32_{off:04x}",
                                offset=off,
                                size=4,
                                kind=FieldKind.I32,
                                sample_values=i_vals,
                                confidence=0.85,
                            )
                        )
                    else:
                        fields.append(
                            StructField(
                                name=f"u32_{off:04x}",
                                offset=off,
                                size=4,
                                kind=FieldKind.U32,
                                sample_values=u_vals,
                                confidence=0.85,
                            )
                        )
                    off += 4
                    continue

        if unclaimed_start is None:
            unclaimed_start = off
        off += 1

    _flush_pad(obj_size)

    struct_align = max((f.alignment for f in fields), default=1)
    final_size = obj_size
    if obj_size % struct_align != 0:
        tail_pad = struct_align - (obj_size % struct_align)
        final_size = obj_size + tail_pad
        if fields and fields[-1].kind == FieldKind.PAD:
            fields[-1].size += tail_pad
        else:
            fields.append(
                StructField(
                    name=f"_pad_{obj_size:04x}",
                    offset=obj_size,
                    size=tail_pad,
                    kind=FieldKind.PAD,
                    confidence=1.0,
                )
            )

    demangled_rtti: str | None = None
    vtable_rva: int | None = None
    base_classes: list[str] = []
    if fields and fields[0].offset == 0 and fields[0].kind == FieldKind.VPTR:
        vtable_rva = fields[0].vtable_rva
        if vtable_rva is not None and vtable_rva in vt_by_rva:
            vt = vt_by_rva[vtable_rva]
            demangled_rtti = vt.demangled_name
            base_classes = list(vt.base_classes)

    return StructDef(
        name=sanitize_ident(name),
        size=final_size,
        alignment=struct_align,
        fields=fields,
        demangled_rtti=demangled_rtti,
        vtable_rva=vtable_rva,
        base_classes=base_classes,
        instance_vas=list(obj_vas),
    )


def _match_watch_at(
    snapshots: Sequence[HeapSnapshot],
    cur_vas: tuple[int, ...],
    leaf_off: int,
    watch: WatchSpec,
) -> bool:
    if len(watch.values) != len(snapshots):
        return False
    fsize = FIXED_FIELD_SIZES.get(watch.kind)
    if fsize is None:
        return False

    for snap, va, expected in zip(snapshots, cur_vas, watch.values):
        raw = snap.read_va(va + leaf_off, fsize)
        if raw is None:
            return False
        if watch.kind == FieldKind.F32:
            val = struct.unpack("<f", raw)[0]
            if not math.isfinite(val) or abs(val - float(expected)) > 1e-4:
                return False
        elif watch.kind == FieldKind.F64:
            val = struct.unpack("<d", raw)[0]
            if not math.isfinite(val) or abs(val - float(expected)) > 1e-4:
                return False
        elif watch.kind == FieldKind.BOOL:
            if raw[0] not in (0, 1) or (raw[0] != 0) != bool(expected):
                return False
        elif watch.kind == FieldKind.I32:
            if struct.unpack("<i", raw)[0] != int(expected):
                return False
        elif watch.kind == FieldKind.U32:
            if struct.unpack("<I", raw)[0] != int(expected):
                return False
        elif watch.kind == FieldKind.I64:
            if struct.unpack("<q", raw)[0] != int(expected):
                return False
        elif watch.kind == FieldKind.U64:
            if struct.unpack("<Q", raw)[0] != int(expected):
                return False
        elif watch.kind == FieldKind.U16:
            if struct.unpack("<H", raw)[0] != int(expected):
                return False
        elif watch.kind == FieldKind.U8:
            if raw[0] != int(expected):
                return False
        elif watch.kind == FieldKind.CSTR_PTR:
            ptr = struct.unpack("<Q", raw)[0]
            if snap.read_cstr_va(ptr) != str(expected):
                return False
        elif watch.kind == FieldKind.STD_VECTOR:
            b, e, c = struct.unpack("<QQQ", raw)
            if not (b != 0 and b <= e <= c and snap.contains_va(b)):
                return False
            if (e - b) != int(expected):
                return False
        else:
            return False
    return True


def _find_pointer_chains_with_nodes(
    watches: Sequence[WatchSpec],
    snapshots: Sequence[HeapSnapshot],
    root_va: int | None = None,
    max_depth: int = 4,
    max_span: int = 0x200,
) -> tuple[list[PointerChain], dict[tuple[int, ...], tuple[int, ...]]]:
    if not watches:
        return [], {}
    if len(snapshots) < 2:
        raise InferenceError("find_pointer_chains requires at least 2 snapshots")

    base_root_va = root_va if root_va is not None else snapshots[0].root_va
    root_rva = base_root_va - snapshots[0].image_base

    p0_list: list[int] = []
    for snap in snapshots:
        raw = snap.read_va(snap.image_base + root_rva, 8)
        if raw is None:
            raise InferenceError(f"Root VA 0x{snap.image_base + root_rva:x} not readable")
        p0 = struct.unpack("<Q", raw)[0]
        if not snap.contains_va(p0, 8):
            raise InferenceError(f"Root pointer 0x{p0:x} is not a valid heap VA")
        p0_list.append(p0)

    p0_tuple = tuple(p0_list)
    queue: deque[tuple[tuple[int, ...], tuple[int, ...]]] = deque([(p0_tuple, ())])
    visited: set[tuple[int, ...]] = {p0_tuple}
    node_map: dict[tuple[int, ...], tuple[int, ...]] = {(): p0_tuple}

    found: dict[str, PointerChain] = {}
    remaining = list(watches)

    while queue and remaining:
        cur_vas, path_offsets = queue.popleft()
        node_map[path_offsets] = cur_vas

        still_remaining: list[WatchSpec] = []
        for watch in remaining:
            align = FIELD_ALIGNMENT[watch.kind]
            matched_off: int | None = None
            for leaf_off in range(0, max_span, align):
                if _match_watch_at(snapshots, cur_vas, leaf_off, watch):
                    matched_off = leaf_off
                    break
            if matched_off is not None:
                found[watch.name] = PointerChain(
                    name=sanitize_ident(watch.name),
                    root_rva=root_rva,
                    offsets=[*path_offsets, matched_off],
                    field_kind=watch.kind,
                    expected_values=list(watch.values),
                )
            else:
                still_remaining.append(watch)
        remaining = still_remaining

        if not remaining:
            break

        if len(path_offsets) + 1 < max_depth:
            for ptr_off in range(0, max_span, 8):
                next_list: list[int] = []
                valid = True
                for snap, va in zip(snapshots, cur_vas):
                    raw = snap.read_va(va + ptr_off, 8)
                    if raw is None:
                        valid = False
                        break
                    nxt = struct.unpack("<Q", raw)[0]
                    if nxt == 0 or not snap.contains_va(nxt, 8):
                        valid = False
                        break
                    next_list.append(nxt)
                if valid:
                    nxt_tuple = tuple(next_list)
                    if nxt_tuple not in visited:
                        visited.add(nxt_tuple)
                        nxt_path = (*path_offsets, ptr_off)
                        node_map[nxt_path] = nxt_tuple
                        queue.append((nxt_tuple, nxt_path))

    missing = [w.name for w in watches if w.name not in found]
    if missing:
        raise InferenceError(f"Could not locate pointer chain for watch(es): {', '.join(missing)}")
    chains = [found[w.name] for w in watches]
    return chains, node_map


def find_pointer_chains(
    watches: Sequence[WatchSpec],
    snapshots: Sequence[HeapSnapshot],
    root_va: int | None = None,
    max_depth: int = 4,
    max_span: int = 0x200,
) -> list[PointerChain]:
    chains, _ = _find_pointer_chains_with_nodes(
        watches=watches,
        snapshots=snapshots,
        root_va=root_va,
        max_depth=max_depth,
        max_span=max_span,
    )
    return chains


def _estimate_node_size(
    node_vas: tuple[int, ...],
    snapshots: Sequence[HeapSnapshot],
    all_node_vas: set[int],
    min_required_end: int,
    max_probe: int = 0x80,
) -> int:
    # Stop probe before the closest other known allocation/node VA in snapshot 0 to avoid bleeding
    probe_limit = max_probe
    va0 = node_vas[0]
    for other_va in all_node_vas:
        if other_va >= va0 + min_required_end and other_va - va0 < probe_limit:
            probe_limit = other_va - va0

    probe_limit = max(probe_limit, min_required_end)

    last_nonzero_end = min_required_end
    for snap, va in zip(snapshots, node_vas):
        reg = snap.region_at_va(va)
        if reg is None:
            continue
        avail = min(probe_limit, len(reg.data) - (va - reg.base_va))
        chunk = reg.data[va - reg.base_va : va - reg.base_va + avail]
        for idx in range(avail - 1, -1, -1):
            if chunk[idx] != 0:
                if idx + 1 > last_nonzero_end:
                    last_nonzero_end = idx + 1
                break

    # Align up to 8 bytes
    aligned = (last_nonzero_end + 7) & ~7
    return max(8, aligned)


def synthesize_sdk(
    snapshots: Sequence[HeapSnapshot],
    binary: LoadedBinary | None = None,
    watches: Sequence[WatchSpec] = (),
    name: str = "GeneratedSdk",
    module_name: str = "module.dll",
    root_va: int | None = None,
    struct_targets: Sequence[tuple[str, Sequence[int], int]] = (),
    max_depth: int = 4,
    max_span: int = 0x200,
) -> tuple[SdkSpec, OracleResult]:
    if not watches and not struct_targets:
        raise InferenceError("synthesize_sdk requires at least one WatchSpec or struct_target")
    vtables = harvest_rtti(binary) if binary is not None else []
    vt_by_rva = {vt.vtable_rva: vt for vt in vtables}

    chains, node_map = _find_pointer_chains_with_nodes(
        watches=watches,
        snapshots=snapshots,
        root_va=root_va,
        max_depth=max_depth,
        max_span=max_span,
    )

    # Identify all active paths referenced by discovered chains
    active_paths: dict[tuple[int, ...], tuple[int, ...]] = {}
    overrides_by_path: dict[tuple[int, ...], dict[int, tuple[str, FieldKind]]] = {}
    min_end_by_path: dict[tuple[int, ...], int] = {}

    for ch in chains:
        parent_path: tuple[int, ...] = ()
        for hop_off in ch.offsets[:-1]:
            if parent_path in node_map:
                active_paths[parent_path] = node_map[parent_path]
                min_end_by_path[parent_path] = max(min_end_by_path.get(parent_path, 0), hop_off + 8)
            parent_path = (*parent_path, hop_off)

        if parent_path in node_map:
            active_paths[parent_path] = node_map[parent_path]
            leaf_off = ch.offsets[-1]
            overrides_by_path.setdefault(parent_path, {})[leaf_off] = (ch.name, ch.field_kind)
            fsize = FIXED_FIELD_SIZES.get(ch.field_kind, 8)
            min_end_by_path[parent_path] = max(
                min_end_by_path.get(parent_path, 0), leaf_off + fsize
            )

    all_node_va0s = {vas[0] for vas in active_paths.values()}
    if snapshots:
        snap0 = snapshots[0]
        for reg in snap0.regions:
            limit = len(reg.data) - (len(reg.data) % 8)
            for off in range(0, limit, 8):
                cand_ptr = struct.unpack_from("<Q", reg.data, off)[0]
                if cand_ptr != 0 and snap0.contains_va(cand_ptr):
                    all_node_va0s.add(cand_ptr)

    structs: list[StructDef] = []
    used_names: set[str] = set()
    path_to_struct_name: dict[tuple[int, ...], str] = {}

    def _unique_name(base: str) -> str:
        cand = sanitize_ident(base)
        if cand not in used_names:
            used_names.add(cand)
            return cand
        idx = 1
        while f"{cand}_{idx}" in used_names:
            idx += 1
        final = f"{cand}_{idx}"
        used_names.add(final)
        return final

    for path in sorted(active_paths, key=lambda p: (len(p), p)):
        node_vas = active_paths[path]
        raw_vptr = snapshots[0].read_va(node_vas[0], 8)
        st_name_base: str | None = None
        if raw_vptr is not None:
            rva = struct.unpack("<Q", raw_vptr)[0] - snapshots[0].image_base
            if rva in vt_by_rva:
                st_name_base = vt_by_rva[rva].demangled_name.split("::")[-1]
        if st_name_base is None:
            st_name_base = "WorldContext" if len(path) == 0 else f"StructNode_{len(path)}"

        st_name = _unique_name(st_name_base)
        path_to_struct_name[path] = st_name
        est_size = _estimate_node_size(
            node_vas=node_vas,
            snapshots=snapshots,
            all_node_vas=all_node_va0s,
            min_required_end=min_end_by_path.get(path, 8),
        )
        st_def = infer_struct(
            name=st_name,
            obj_vas=node_vas,
            obj_size=est_size,
            snapshots=snapshots,
            vtables=vtables,
            field_overrides=overrides_by_path.get(path),
        )
        structs.append(st_def)

    for ch in chains:
        leaf_path = tuple(ch.offsets[:-1])
        if leaf_path in path_to_struct_name:
            ch.target_struct = path_to_struct_name[leaf_path]

    for t_name, t_vas, t_size in struct_targets:
        st_name = _unique_name(t_name)
        st_def = infer_struct(
            name=st_name,
            obj_vas=t_vas,
            obj_size=t_size,
            snapshots=snapshots,
            vtables=vtables,
        )
        structs.append(st_def)

    fmt = binary.format if binary is not None else "raw64"
    spec = SdkSpec(
        name=sanitize_ident(name),
        binary_format=fmt,
        module_name=module_name,
        structs=structs,
        vtables=vtables,
        pointer_chains=chains,
        description=f"Synthesized {len(structs)} structs and {len(chains)} pointer chains",
    )
    validate_spec(spec)
    oracle = verify_sdk(spec, snapshots, binary=binary)
    spec.layout_score = oracle.layout_score
    return spec, oracle
