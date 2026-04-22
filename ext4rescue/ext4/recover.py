from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .super import SUPERBLOCK_OFFSET, backup_superblock_offsets, parse_superblock
from .gdt import inode_physical_offset, read_gdt
from .inode import Ext4Inode, parse_inode
from .extent import parse_extent_tree
from .dir import parse_directory_block
from ..utils import safe_filename, safe_path_components

log = logging.getLogger(__name__)


@dataclass
class RecoveryResult:
    named_count: int = 0
    orphan_count: int = 0
    error_count: int = 0
    output_dir: str = ""
    session_json: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "named_count": self.named_count,
            "orphan_count": self.orphan_count,
            "error_count": self.error_count,
            "output_dir": self.output_dir,
            "session_json": self.session_json,
            "warnings": self.warnings,
        }


class DiskImage:
    def __init__(self, path: str) -> None:
        self.path = path
        self.fh = open(path, "rb", buffering=0)
        self.fd = self.fh.fileno()
        self.size = os.path.getsize(path)

    def close(self) -> None:
        self.fh.close()

    def read_at(self, offset: int, size: int) -> bytes:
        return os.pread(self.fd, size, offset)

    def read_block(self, block_nr: int, block_size: int) -> bytes | None:
        off = block_nr * block_size
        if off < 0 or off >= self.size:
            return None
        data = self.read_at(off, block_size)
        return data if len(data) == block_size else None


class _DictObj:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


def _sparse_groups(limit: int) -> list[int]:
    groups = [g for g in range(1, limit) if _is_sparse_group(g)]
    return groups


def _is_sparse_group(group: int) -> bool:
    if group <= 1:
        return True
    if group % 2 == 0:
        return False
    for base in (3, 5, 7):
        n = base
        while n < group:
            n *= base
        if n == group:
            return True
    return False


def _find_best_superblock(disk: DiskImage, sb_offset: int | None = None):
    candidates = []
    if sb_offset is not None:
        buf = disk.read_at(sb_offset, 1024)
        sb = parse_superblock(buf, raw_offset=sb_offset)
        if sb.is_valid:
            return sb
        raise ValueError(f"Invalid superblock at explicit offset {sb_offset}")

    primary = parse_superblock(disk.read_at(SUPERBLOCK_OFFSET, 1024), raw_offset=SUPERBLOCK_OFFSET)
    if primary.is_valid:
        candidates.append(primary)
        for off in backup_superblock_offsets(primary.block_size, primary.blocks_per_group, min(primary.total_groups, 256)):
            if off + 1024 <= disk.size:
                sb = parse_superblock(disk.read_at(off, 1024), raw_offset=off)
                if sb.is_valid:
                    candidates.append(sb)
    else:
        group_size = 128 * 1024 * 1024
        max_groups = min(disk.size // group_size, 256)
        for g in _sparse_groups(int(max_groups)):
            off = g * group_size
            if off + 1024 > disk.size:
                break
            sb = parse_superblock(disk.read_at(off, 1024), raw_offset=off)
            if sb.is_valid:
                candidates.append(sb)

    if not candidates:
        raise ValueError("No valid ext4 superblock found")
    candidates.sort(key=lambda sb: (sb.score, sb.blocks_count), reverse=True)
    return candidates[0]


def _read_inode(disk: DiskImage, sb, gdt, inode_nr: int) -> Ext4Inode | None:
    off = inode_physical_offset(inode_nr, gdt, sb)
    if off is None:
        return None
    buf = disk.read_at(off, sb.inode_size)
    if len(buf) < sb.inode_size:
        return None
    ino = parse_inode(buf, inode_nr=inode_nr)
    return ino if ino.is_valid else None


def _inode_block_numbers(disk: DiskImage, sb, inode: Ext4Inode) -> list[int]:
    if inode.uses_extents:
        result = parse_extent_tree(
            inode.i_block,
            read_block_fn=lambda b: disk.read_block(b, sb.block_size),
            disk_blocks=sb.blocks_count,
        )
        blocks: list[int] = []
        for ext in result.extents:
            for i in range(ext.length):
                blocks.append(ext.physical_block + i)
        return blocks
    return inode.direct_blocks


def _read_inode_bytes(disk: DiskImage, sb, inode: Ext4Inode) -> bytes:
    blocks = _inode_block_numbers(disk, sb, inode)
    chunks: list[bytes] = []
    remaining = inode.size
    for block_nr in blocks:
        if remaining <= 0:
            break
        buf = disk.read_block(block_nr, sb.block_size)
        if not buf:
            break
        take = min(len(buf), remaining)
        chunks.append(buf[:take])
        remaining -= take
    return b"".join(chunks)


def _iter_dir_entries(disk: DiskImage, sb, inode: Ext4Inode):
    raw = _read_inode_bytes(disk, sb, inode)
    if not raw:
        return []
    entries = []
    for off in range(0, len(raw), sb.block_size):
        entries.extend(parse_directory_block(raw[off:off + sb.block_size]))
    return entries


def _write_file(base_dir: str, rel_components: list[str], name: str, data: bytes) -> str:
    safe_parts = safe_path_components(rel_components)
    safe_name = safe_filename(name)
    out_dir = os.path.join(base_dir, *safe_parts) if safe_parts else base_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, safe_name)
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def run_recovery(
    disk_path: str,
    output_dir: str,
    *,
    sb_offset: int | None = None,
    extract_orphans: bool = True,
    max_orphans: int = 256,
) -> RecoveryResult:
    result = RecoveryResult(output_dir=output_dir)
    named_root = os.path.join(output_dir, "named")
    orphan_root = os.path.join(output_dir, "_orphans")
    os.makedirs(named_root, exist_ok=True)
    os.makedirs(orphan_root, exist_ok=True)

    disk = DiskImage(disk_path)
    visited_files: set[int] = set()
    visited_dirs: set[int] = set()
    extracted_orphans = 0

    try:
        sb = _find_best_superblock(disk, sb_offset=sb_offset)
        gdt = read_gdt(disk, sb)

        def walk_dir(dir_inode_nr: int, rel_path: list[str]) -> None:
            if dir_inode_nr in visited_dirs:
                return
            visited_dirs.add(dir_inode_nr)
            inode = _read_inode(disk, sb, gdt, dir_inode_nr)
            if inode is None or not inode.is_dir:
                return
            for entry in _iter_dir_entries(disk, sb, inode):
                if not entry.is_valid or entry.name in {".", ".."}:
                    continue
                child = _read_inode(disk, sb, gdt, entry.inode_nr)
                if child is None:
                    result.error_count += 1
                    continue
                if child.is_dir:
                    walk_dir(entry.inode_nr, rel_path + [entry.name])
                elif child.is_regular:
                    try:
                        data = _read_inode_bytes(disk, sb, child)
                        _write_file(named_root, rel_path, entry.name, data)
                        visited_files.add(entry.inode_nr)
                        result.named_count += 1
                    except OSError as exc:
                        result.error_count += 1
                        result.warnings.append(f"Failed extracting inode {entry.inode_nr}: {exc}")

        walk_dir(2, [])

        if extract_orphans:
            total_inodes = min(sb.inodes_count, sb.inodes_per_group * len(gdt))
            for inode_nr in range(1, total_inodes + 1):
                if extracted_orphans >= max_orphans:
                    result.warnings.append(f"Orphan extraction capped at {max_orphans} files.")
                    break
                if inode_nr in visited_files or inode_nr in visited_dirs:
                    continue
                inode = _read_inode(disk, sb, gdt, inode_nr)
                if inode is None or not inode.is_regular:
                    continue
                try:
                    data = _read_inode_bytes(disk, sb, inode)
                    ext = ".bin"
                    name = f"inode_{inode_nr}{ext}"
                    _write_file(orphan_root, [], name, data)
                    result.orphan_count += 1
                    extracted_orphans += 1
                except OSError as exc:
                    result.error_count += 1
                    result.warnings.append(f"Failed orphan extraction inode {inode_nr}: {exc}")

        session = {
            "disk_path": disk_path,
            "superblock_offset": sb.raw_offset,
            "block_size": sb.block_size,
            "named_count": result.named_count,
            "orphan_count": result.orphan_count,
            "error_count": result.error_count,
            "warnings": result.warnings,
        }
        session_json = os.path.join(output_dir, "session.json")
        with open(session_json, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        result.session_json = session_json
        return result
    finally:
        disk.close()
