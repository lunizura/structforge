from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from structforge.errors import SnapshotFormatError

_SFMD_MAGIC = b"SFMD"
_SFMD_VERSION = 1
_HEADER_FMT = "<4sIIQQI"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_REGION_HDR_FMT = "<QQII"
_REGION_HDR_SIZE = struct.calcsize(_REGION_HDR_FMT)


@dataclass
class MemoryRegion:
    name: str
    base_va: int
    data: bytes
    flags: int = 1


@dataclass
class HeapSnapshot:
    image_base: int
    root_va: int
    regions: list[MemoryRegion]

    def region_at_va(self, va: int) -> MemoryRegion | None:
        for reg in self.regions:
            if reg.base_va <= va < reg.base_va + len(reg.data):
                return reg
        return None

    def contains_va(self, va: int, size: int = 1) -> bool:
        if size <= 0 or va < 0:
            return False
        reg = self.region_at_va(va)
        if reg is None:
            return False
        off = va - reg.base_va
        return off + size <= len(reg.data)

    def read_va(self, va: int, size: int) -> bytes | None:
        if size <= 0 or va < 0:
            return None
        reg = self.region_at_va(va)
        if reg is None:
            return None
        off = va - reg.base_va
        if off + size > len(reg.data):
            return None
        return reg.data[off : off + size]

    def read_cstr_va(self, va: int, max_len: int = 64) -> str | None:
        if va < 0 or max_len <= 0:
            return None
        reg = self.region_at_va(va)
        if reg is None:
            return None
        off = va - reg.base_va
        chunk = reg.data[off : min(len(reg.data), off + max_len)]
        nul = chunk.find(b"\x00")
        if nul < 3:
            return None
        raw = chunk[:nul]
        if not all(0x20 <= b <= 0x7E for b in raw):
            return None
        return raw.decode("ascii")

    def relocate_clone(
        self, image_delta: int, heap_delta: int, vtable_rvas: set[int]
    ) -> HeapSnapshot:
        old_image_base = self.image_base
        global_ranges: list[tuple[int, int]] = []
        heap_ranges: list[tuple[int, int]] = []
        for reg in self.regions:
            lo = reg.base_va
            hi = reg.base_va + len(reg.data)
            if reg.flags & 2:
                global_ranges.append((lo, hi))
            elif reg.flags & 1:
                heap_ranges.append((lo, hi))

        def _in_global(val: int) -> bool:
            return any(lo <= val < hi for lo, hi in global_ranges)

        def _in_heap(val: int) -> bool:
            return any(lo <= val <= hi for lo, hi in heap_ranges)

        new_regions: list[MemoryRegion] = []
        for reg in self.regions:
            buf = bytearray(reg.data)
            limit = len(buf) - (len(buf) % 8)
            for off in range(0, limit, 8):
                val = struct.unpack_from("<Q", buf, off)[0]
                if val == 0:
                    continue
                if _in_heap(val):
                    struct.pack_into("<Q", buf, off, val + heap_delta)
                elif (val - old_image_base) in vtable_rvas or _in_global(val):
                    struct.pack_into("<Q", buf, off, val + image_delta)

            if reg.flags & 2:
                new_base = reg.base_va + image_delta
            elif reg.flags & 1:
                new_base = reg.base_va + heap_delta
            else:
                new_base = reg.base_va

            new_regions.append(
                MemoryRegion(
                    name=reg.name,
                    base_va=new_base,
                    data=bytes(buf),
                    flags=reg.flags,
                )
            )

        root_delta = (
            image_delta
            if _in_global(self.root_va)
            else (heap_delta if _in_heap(self.root_va) else 0)
        )
        new_root = self.root_va + root_delta
        return HeapSnapshot(
            image_base=old_image_base + image_delta,
            root_va=new_root,
            regions=new_regions,
        )


def serialize_snapshot(snap: HeapSnapshot) -> bytes:
    out = bytearray(
        struct.pack(
            _HEADER_FMT,
            _SFMD_MAGIC,
            _SFMD_VERSION,
            len(snap.regions),
            snap.image_base,
            snap.root_va,
            0,
        )
    )
    for reg in snap.regions:
        name_bytes = reg.name.encode("utf-8")
        out.extend(
            struct.pack(
                _REGION_HDR_FMT,
                reg.base_va,
                len(reg.data),
                reg.flags,
                len(name_bytes),
            )
        )
        out.extend(name_bytes)
        out.extend(reg.data)
    return bytes(out)


def parse_snapshot(raw: bytes) -> HeapSnapshot:
    if len(raw) < _HEADER_SIZE:
        raise SnapshotFormatError("Snapshot file smaller than 32-byte SFMD header")
    magic, version, num_regions, image_base, root_va, _reserved = struct.unpack_from(
        _HEADER_FMT, raw, 0
    )
    if magic != _SFMD_MAGIC:
        raise SnapshotFormatError(f"Invalid snapshot magic {magic!r} (expected b'SFMD')")
    if version != _SFMD_VERSION:
        raise SnapshotFormatError(f"Unsupported snapshot version {version}")

    pos = _HEADER_SIZE
    regions: list[MemoryRegion] = []
    for idx in range(num_regions):
        if pos + _REGION_HDR_SIZE > len(raw):
            raise SnapshotFormatError(f"Truncated region header at index {idx}")
        base_va, data_len, flags, name_len = struct.unpack_from(_REGION_HDR_FMT, raw, pos)
        pos += _REGION_HDR_SIZE
        if pos + name_len + data_len > len(raw):
            raise SnapshotFormatError(f"Truncated region payload at index {idx}")
        name_bytes = raw[pos : pos + name_len]
        pos += name_len
        try:
            name = name_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SnapshotFormatError(f"Invalid UTF-8 region name at index {idx}") from exc
        data = raw[pos : pos + data_len]
        pos += data_len
        regions.append(MemoryRegion(name=name, base_va=base_va, data=data, flags=flags))

    if pos != len(raw):
        raise SnapshotFormatError(f"Trailing {len(raw) - pos} unexpected bytes in snapshot")

    return HeapSnapshot(image_base=image_base, root_va=root_va, regions=regions)


def save_snapshot(snap: HeapSnapshot, path: str | Path) -> None:
    Path(path).write_bytes(serialize_snapshot(snap))


def load_snapshot(path: str | Path) -> HeapSnapshot:
    return parse_snapshot(Path(path).read_bytes())
