"""
ext4rescue/ext4/dir.py — ext4 directory entry parser and path reconstructor.

Parses ``ext4_dir_entry_2`` records from raw directory block bytes, with
optional HTree (htree / dx_root) indexed directory support.  Reconstructs
file paths by following parent inode links, with cycle detection and depth
capping to prevent infinite loops on corrupted metadata.

Names **always** come from directory entries, never from inode metadata.
Entries whose path cannot be reconstructed are marked as orphans.

All on-disk integers are **little-endian**.

Reference: Linux ``fs/ext4/dir.c``, ``fs/ext4/namei.c``.
"""

from __future__ import annotations

import struct
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DIR_ENTRY_MIN: int = 8
"""Minimum size of an ext4_dir_entry_2 (inode + rec_len + name_len + file_type)."""

_MAX_NAME_LEN: int = 255

_MAX_RECONSTRUCT_DEPTH: int = 64
"""Cap on path-reconstruction recursion to prevent infinite loops."""

# HTree magic values
_DX_ROOT_INFO_RESERVED: int = 0        # reserved field must be 0
_HTREE_HASH_HALF_MD4: int = 0x01
_HTREE_HASH_TEA: int = 0x02

# ext4 file type byte values (from the dir entry, not the inode mode)
FILE_TYPE_UNKNOWN: int = 0
FILE_TYPE_REG: int = 1
FILE_TYPE_DIR: int = 2
FILE_TYPE_CHR: int = 3
FILE_TYPE_BLK: int = 4
FILE_TYPE_FIFO: int = 5
FILE_TYPE_SOCK: int = 6
FILE_TYPE_LNK: int = 7

_FILE_TYPE_NAMES: dict[int, str] = {
    0: "unknown", 1: "regular", 2: "directory",
    3: "char_dev", 4: "block_dev", 5: "fifo",
    6: "socket",  7: "symlink",
}

# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class DirEntry:
    """A single parsed ``ext4_dir_entry_2`` record."""

    inode_nr: int = 0
    name: str = ""
    file_type: int = 0
    is_valid: bool = False

    @property
    def file_type_name(self) -> str:
        return _FILE_TYPE_NAMES.get(self.file_type, "unknown")

    def to_dict(self) -> dict[str, Any]:
        return {
            "inode_nr":       self.inode_nr,
            "name":           self.name,
            "file_type":      self.file_type,
            "file_type_name": self.file_type_name,
            "is_valid":       self.is_valid,
        }


@dataclass
class ReconstructedPath:
    """Result of a path-reconstruction attempt for a single inode."""

    inode_nr: int = 0
    path: str = ""
    is_orphan: bool = False
    """True when reconstruction failed (cycle, depth cap, missing parent)."""

    reason: str = ""
    """Human-readable explanation when ``is_orphan`` is True."""


# ── Public API ────────────────────────────────────────────────────────────────

def parse_directory_block(buf: bytes) -> list[DirEntry]:
    """
    Parse a raw ext4 directory block as a list of ``ext4_dir_entry_2`` entries.

    Malformed entries are skipped with a debug log message rather than raising
    an exception.  Parsing stops when the buffer is exhausted or an entry with
    ``rec_len == 0`` is encountered.

    Args:
        buf: Raw bytes of one ext4 directory block.

    Returns:
        List of :class:`DirEntry` objects (inode 0 entries are excluded).
    """
    entries: list[DirEntry] = []
    offset = 0

    while offset + _DIR_ENTRY_MIN <= len(buf):
        try:
            inode_nr = struct.unpack_from("<I", buf, offset)[0]
            rec_len  = struct.unpack_from("<H", buf, offset + 4)[0]
            name_len = struct.unpack_from("<B", buf, offset + 6)[0]
            file_type = struct.unpack_from("<B", buf, offset + 7)[0]
        except struct.error as exc:
            log.debug("parse_directory_block: struct error at offset %d: %s", offset, exc)
            break

        if rec_len == 0:
            log.debug("parse_directory_block: rec_len=0 at offset %d — stopping", offset)
            break

        if rec_len < _DIR_ENTRY_MIN:
            log.debug(
                "parse_directory_block: rec_len=%d < min at offset %d — skipping 4 bytes",
                rec_len, offset,
            )
            offset += 4   # minimal forward progress to avoid infinite loop
            continue

        if offset + rec_len > len(buf):
            log.debug(
                "parse_directory_block: rec_len=%d would exceed buffer at offset %d",
                rec_len, offset,
            )
            break

        if inode_nr != 0 and 0 < name_len <= _MAX_NAME_LEN:
            name_end = offset + 8 + name_len
            if name_end <= len(buf):
                raw_name = buf[offset + 8: name_end]
                name = _decode_name(raw_name)
                name = _sanitize_name(name)
                entries.append(DirEntry(
                    inode_nr=inode_nr,
                    name=name,
                    file_type=file_type,
                    is_valid=True,
                ))
            else:
                log.debug(
                    "parse_directory_block: name extends beyond buffer at offset %d",
                    offset,
                )

        offset += rec_len

    return entries


def parse_htree_root(buf: bytes) -> dict[str, Any] | None:
    """
    Parse an HTree (dx_root) indexed directory root block.

    An HTree root block begins with the standard ``'.'`` and ``'..'`` entries
    followed by a ``dx_root_info`` structure.  We extract the tree parameters
    (hash version, tree depth, indirect levels) without attempting a full
    B-tree traversal.

    Args:
        buf: Raw bytes of the block (typically one filesystem block).

    Returns:
        Dict with HTree metadata or ``None`` when the block does not look like
        a valid HTree root.
    """
    if len(buf) < 40:
        return None

    # The first two entries are '.' and '..'; validate them minimally
    try:
        dot_inode = struct.unpack_from("<I", buf, 0)[0]
        dot_reclen = struct.unpack_from("<H", buf, 4)[0]
        dotdot_inode = struct.unpack_from("<I", buf, dot_reclen)[0]
    except struct.error:
        return None

    if dot_inode == 0 or dot_reclen < _DIR_ENTRY_MIN:
        return None

    # dx_root_info starts at: dot_reclen + 8 (skip dotdot's inode/rec_len/etc.)
    info_offset = dot_reclen + 8
    if info_offset + 8 > len(buf):
        return None

    try:
        # struct dx_root_info {
        #   __le32  reserved_zero;
        #   __u8    hash_version;
        #   __u8    info_length;
        #   __u8    indirect_levels;
        #   __u8    unused_flags;
        # }
        reserved_zero = struct.unpack_from("<I", buf, info_offset)[0]
        hash_version  = struct.unpack_from("<B", buf, info_offset + 4)[0]
        info_length   = struct.unpack_from("<B", buf, info_offset + 5)[0]
        indirect_levels = struct.unpack_from("<B", buf, info_offset + 6)[0]
    except struct.error:
        return None

    if reserved_zero != 0:
        log.debug("parse_htree_root: reserved_zero = %d (expected 0)", reserved_zero)
        return None   # Not an HTree root

    return {
        "dot_inode":      dot_inode,
        "dotdot_inode":   dotdot_inode,
        "hash_version":   hash_version,
        "indirect_levels": indirect_levels,
        "info_length":    info_length,
        "is_htree":       True,
    }


def reconstruct_path(
    inode_nr: int,
    get_entries_fn: Callable[[int], list[DirEntry]],
    root_inode: int = 2,
) -> ReconstructedPath:
    """
    Reconstruct the absolute path of *inode_nr* by walking parent directories.

    The caller supplies *get_entries_fn* which, given a directory inode number,
    returns the parsed :class:`DirEntry` list for that directory (or an empty
    list on failure).  This design keeps I/O outside this module.

    Algorithm:
    1. Look up *inode_nr* in its parent directory to find its name.
    2. Follow the ``'..'`` entries upward until we reach *root_inode* (inode 2)
       or exhaust the depth cap.
    3. Detect cycles with a ``visited`` set.

    Args:
        inode_nr:       Inode to resolve.
        get_entries_fn: Callable returning directory entries for a given inode.
        root_inode:     Inode number of the filesystem root (default: 2).

    Returns:
        :class:`ReconstructedPath` with ``path`` set to a ``/``-separated
        absolute path, or ``is_orphan = True`` with a reason on failure.
    """
    components: list[str] = []
    visited: set[int] = set()
    current = inode_nr

    for _depth in range(_MAX_RECONSTRUCT_DEPTH):
        if current in visited:
            return ReconstructedPath(
                inode_nr=inode_nr,
                is_orphan=True,
                reason=f"cycle detected at inode {current}",
            )
        visited.add(current)

        if current == root_inode:
            break

        # Find which directory contains `current` and what name it has
        parent_inode, name = _find_in_parent(current, get_entries_fn)
        if parent_inode is None:
            return ReconstructedPath(
                inode_nr=inode_nr,
                is_orphan=True,
                reason=f"could not find parent of inode {current}",
            )

        if name is None:
            return ReconstructedPath(
                inode_nr=inode_nr,
                is_orphan=True,
                reason=f"inode {current} found in parent {parent_inode} but name missing",
            )

        # Self-parent guard
        if parent_inode == current:
            return ReconstructedPath(
                inode_nr=inode_nr,
                is_orphan=True,
                reason=f"inode {current} is its own parent — corrupt directory",
            )

        components.append(name)
        current = parent_inode

    else:
        return ReconstructedPath(
            inode_nr=inode_nr,
            is_orphan=True,
            reason=f"depth cap ({_MAX_RECONSTRUCT_DEPTH}) reached",
        )

    path = "/" + "/".join(reversed(components)) if components else "/"
    return ReconstructedPath(inode_nr=inode_nr, path=path)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_in_parent(
    inode_nr: int,
    get_entries_fn: Callable[[int], list[DirEntry]],
) -> tuple[int | None, str | None]:
    """
    Find the name of *inode_nr* and its parent inode by reading the parent
    directory.

    We first try to get the ``'..'`` entry of *inode_nr* itself to find the
    parent, then scan the parent for an entry that references *inode_nr*.

    Returns ``(parent_inode_nr, name)`` or ``(None, None)`` on failure.
    """
    # Step 1: read the entries of `inode_nr` itself to get '..'
    own_entries = get_entries_fn(inode_nr)
    parent_inode: int | None = None
    for e in own_entries:
        if e.name == "..":
            parent_inode = e.inode_nr
            break

    if parent_inode is None:
        return None, None

    # Step 2: scan the parent directory for the entry pointing to inode_nr
    parent_entries = get_entries_fn(parent_inode)
    for e in parent_entries:
        if e.inode_nr == inode_nr and e.name not in {".", ".."}:
            return parent_inode, e.name

    return None, None


def _decode_name(raw: bytes) -> str:
    """Decode a raw filename; try UTF-8 then fall back to Latin-1."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _sanitize_name(name: str) -> str:
    """
    Replace NUL bytes and control characters in a directory entry name.

    Preserves printable ASCII and valid UTF-8; replaces everything else with
    ``'_'``.
    """
    result: list[str] = []
    for ch in name:
        cp = ord(ch)
        if cp == 0 or (cp < 0x20 and cp != 0x09):   # allow tab (0x09)
            result.append("_")
        elif ch == "/":
            result.append("_")
        else:
            result.append(ch)
    return "".join(result)
