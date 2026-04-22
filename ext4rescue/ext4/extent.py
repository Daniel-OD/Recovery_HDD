"""
ext4rescue/ext4/extent.py — ext4 extent tree traversal.

Parses the extent tree embedded in an inode's ``i_block`` field (or in
external tree blocks) and returns a flat list of logical→physical block
mappings.

Supports depth 0 (leaf-only inode) through depth 5 (four levels of index
nodes above leaves).  Corrupted branches are skipped with a warning; the
caller receives whatever mappings could be recovered from healthy branches.

All on-disk integers are **little-endian**.

Reference: Linux ``fs/ext4/ext4_extents.h``.
"""

from __future__ import annotations

import struct
import logging
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EXT4_EXTENT_MAGIC: int = 0xF30A
"""Magic number in every extent header (LE u16)."""

_EXTENT_HEADER_SIZE: int = 12    # sizeof(ext4_extent_header)
_EXTENT_ENTRY_SIZE: int = 12     # sizeof(ext4_extent)   = sizeof(ext4_extent_idx)

_MAX_DEPTH: int = 5
"""Maximum supported tree depth (Linux supports up to 5)."""

_UNINIT_MASK: int = 0x8000
"""ee_len high bit set → uninitialized / pre-allocated extent."""

_MAX_EXTENT_LEN: int = 32768    # uninit threshold; len ≥ this = uninitialized


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ExtentEntry:
    """A single logical→physical block mapping from the extent tree."""

    logical_block: int = 0
    """First logical block covered by this extent."""

    physical_block: int = 0
    """First physical block on disk."""

    length: int = 0
    """Number of blocks in this extent (always ≥ 1 for valid entries)."""

    uninitialized: bool = False
    """True when the extent is pre-allocated but not yet written."""

    def __repr__(self) -> str:
        status = " [uninit]" if self.uninitialized else ""
        return (
            f"ExtentEntry(lba={self.logical_block}, "
            f"pba={self.physical_block}, len={self.length}{status})"
        )


@dataclass
class SparseHole:
    """A sparse hole between two extents (logical blocks with no physical data)."""

    logical_start: int = 0
    logical_end: int = 0     # exclusive


@dataclass
class ExtentTreeResult:
    """Full result of an extent tree traversal."""

    extents: list[ExtentEntry] = field(default_factory=list)
    sparse_holes: list[SparseHole] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_partial: bool = False
    """True when one or more branches were skipped due to corruption."""


# ── Public API ────────────────────────────────────────────────────────────────

def parse_extent_tree(
    iblock: bytes,
    read_block_fn: Callable[[int], bytes | None],
    disk_blocks: int = 0,
) -> ExtentTreeResult:
    """
    Traverse the extent tree rooted at *iblock* (the inode's ``i_block``).

    Args:
        iblock:        60-byte ``i_block`` field from the inode (or any buffer
                       that starts with a valid extent header).
        read_block_fn: Callable ``(phys_block_nr) -> bytes | None`` that reads
                       one filesystem block.  Returns ``None`` on I/O error.
        disk_blocks:   Total number of blocks on the device; used to validate
                       physical block numbers.  Pass 0 to skip bounds checking.

    Returns:
        :class:`ExtentTreeResult` with all recovered extents, detected sparse
        holes, and any warnings generated during traversal.
    """
    result = ExtentTreeResult()

    if len(iblock) < _EXTENT_HEADER_SIZE:
        result.warnings.append(
            f"i_block too short: {len(iblock)} bytes (need {_EXTENT_HEADER_SIZE})"
        )
        return result

    _traverse(iblock, depth_remaining=_MAX_DEPTH,
              result=result, read_block_fn=read_block_fn,
              disk_blocks=disk_blocks, path="iblock")

    # Sort extents by logical block and inject sparse holes
    result.extents.sort(key=lambda e: e.logical_block)
    result.sparse_holes = _find_sparse_holes(result.extents)
    _detect_logical_overlaps(result)
    return result


# ── Internal traversal ────────────────────────────────────────────────────────

def _traverse(
    buf: bytes,
    depth_remaining: int,
    result: ExtentTreeResult,
    read_block_fn: Callable[[int], bytes | None],
    disk_blocks: int,
    path: str,
) -> None:
    """Recursively traverse one extent tree node."""
    hdr = _parse_header(buf, path)
    if hdr is None:
        result.warnings.append(f"{path}: invalid header — branch skipped")
        result.is_partial = True
        return

    magic, eh_entries, eh_max, eh_depth, _gen = hdr

    if eh_depth == 0:
        # Leaf node — parse ext4_extent entries
        _parse_leaf(buf, eh_entries, result, disk_blocks, path)
    else:
        # Index node — recurse into each child
        if depth_remaining <= 0:
            result.warnings.append(
                f"{path}: depth limit exceeded (eh_depth={eh_depth}) — branch skipped"
            )
            result.is_partial = True
            return
        _parse_index(buf, eh_entries, result, read_block_fn,
                     disk_blocks, depth_remaining - 1, path)


def _parse_header(
    buf: bytes, path: str
) -> tuple[int, int, int, int, int] | None:
    """
    Parse an ``ext4_extent_header`` at the start of *buf*.

    Returns ``(magic, entries, max_entries, depth, generation)`` or ``None``.
    """
    if len(buf) < _EXTENT_HEADER_SIZE:
        return None
    try:
        magic, entries, max_entries, depth, gen = struct.unpack_from(
            "<HHHHI", buf, 0
        )
    except struct.error:
        return None

    if magic != EXT4_EXTENT_MAGIC:
        log.debug("%s: bad extent magic 0x%04X (expected 0x%04X)", path, magic, EXT4_EXTENT_MAGIC)
        return None

    if entries > max_entries and max_entries != 0:
        log.debug("%s: entries=%d > max_entries=%d", path, entries, max_entries)
        # Non-fatal; clamp to available space
        entries = min(entries, (len(buf) - _EXTENT_HEADER_SIZE) // _EXTENT_ENTRY_SIZE)

    if depth > _MAX_DEPTH:
        log.debug("%s: implausible depth %d", path, depth)
        return None

    return magic, entries, max_entries, depth, gen


def _parse_leaf(
    buf: bytes,
    num_entries: int,
    result: ExtentTreeResult,
    disk_blocks: int,
    path: str,
) -> None:
    """Parse leaf-level ``ext4_extent`` entries."""
    for i in range(num_entries):
        off = _EXTENT_HEADER_SIZE + i * _EXTENT_ENTRY_SIZE
        if off + _EXTENT_ENTRY_SIZE > len(buf):
            result.warnings.append(
                f"{path}: leaf entry {i} out of bounds (buf={len(buf)})"
            )
            result.is_partial = True
            break

        try:
            ee_block, ee_len, ee_start_hi, ee_start_lo = struct.unpack_from(
                "<IHHI", buf, off
            )
        except struct.error as exc:
            result.warnings.append(f"{path}: struct error at leaf entry {i}: {exc}")
            result.is_partial = True
            break

        uninitialized = bool(ee_len & _UNINIT_MASK)
        real_len = ee_len & ~_UNINIT_MASK

        if real_len == 0:
            result.warnings.append(
                f"{path}: leaf entry {i}: ee_len=0, skipped"
            )
            continue

        phys_block = (ee_start_hi << 32) | ee_start_lo

        if disk_blocks > 0 and phys_block + real_len > disk_blocks:
            result.warnings.append(
                f"{path}: leaf entry {i}: physical block {phys_block}+{real_len} "
                f"exceeds disk size {disk_blocks} — skipped"
            )
            result.is_partial = True
            continue

        result.extents.append(ExtentEntry(
            logical_block=ee_block,
            physical_block=phys_block,
            length=real_len,
            uninitialized=uninitialized,
        ))


def _parse_index(
    buf: bytes,
    num_entries: int,
    result: ExtentTreeResult,
    read_block_fn: Callable[[int], bytes | None],
    disk_blocks: int,
    depth_remaining: int,
    path: str,
) -> None:
    """Parse index-level ``ext4_extent_idx`` entries and recurse."""
    for i in range(num_entries):
        off = _EXTENT_HEADER_SIZE + i * _EXTENT_ENTRY_SIZE
        if off + _EXTENT_ENTRY_SIZE > len(buf):
            result.warnings.append(
                f"{path}: index entry {i} out of bounds (buf={len(buf)})"
            )
            result.is_partial = True
            break

        try:
            # ext4_extent_idx: ei_block, ei_leaf_lo, ei_unused (padding), ei_leaf_hi
            ei_block, ei_leaf_lo, _ei_unused, ei_leaf_hi = struct.unpack_from(
                "<IIHI", buf, off
            )
        except struct.error as exc:
            result.warnings.append(
                f"{path}: struct error at index entry {i}: {exc}"
            )
            result.is_partial = True
            break

        leaf_phys = (ei_leaf_hi << 32) | ei_leaf_lo

        if disk_blocks > 0 and leaf_phys >= disk_blocks:
            result.warnings.append(
                f"{path}: index entry {i}: leaf block {leaf_phys} "
                f"exceeds disk size {disk_blocks} — branch skipped"
            )
            result.is_partial = True
            continue

        child_buf = read_block_fn(leaf_phys)
        if child_buf is None:
            result.warnings.append(
                f"{path}: I/O error reading index child block {leaf_phys} — branch skipped"
            )
            result.is_partial = True
            continue

        child_path = f"{path}→idx[{i}]@{leaf_phys}"
        _traverse(child_buf, depth_remaining, result, read_block_fn,
                  disk_blocks, child_path)


# ── Sparse hole detection ─────────────────────────────────────────────────────

def _find_sparse_holes(extents: list[ExtentEntry]) -> list[SparseHole]:
    """
    Identify gaps between consecutive extents as sparse holes.

    *extents* must already be sorted by ``logical_block``.
    """
    holes: list[SparseHole] = []
    prev_end = 0
    for ext in extents:
        if ext.logical_block > prev_end:
            holes.append(SparseHole(
                logical_start=prev_end,
                logical_end=ext.logical_block,
            ))
        prev_end = max(prev_end, ext.logical_block + ext.length)
    return holes


def _detect_logical_overlaps(result: ExtentTreeResult) -> None:
    """
    Annotate overlapping logical block ranges in the result warnings.

    Overlaps indicate corruption; we keep both extents but log the anomaly.
    """
    prev: ExtentEntry | None = None
    for ext in result.extents:
        if prev is not None:
            prev_end = prev.logical_block + prev.length
            if ext.logical_block < prev_end:
                result.warnings.append(
                    f"Logical overlap: extent at lba={ext.logical_block} "
                    f"overlaps previous extent ending at lba={prev_end}"
                )
                result.is_partial = True
        prev = ext
