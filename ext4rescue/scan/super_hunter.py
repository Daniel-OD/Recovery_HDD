"""
ext4rescue/scan/super_hunter.py — Superblock location scanner.

Scans a disk image for valid ext4 superblocks (primary and backup) and
returns scored candidates.
"""

from __future__ import annotations

import os
import struct
import logging
from typing import Any

from ..ext4.super import (
    EXT4_SUPER_MAGIC,
    SUPERBLOCK_OFFSET,
    parse_superblock,
    backup_superblock_offsets,
)

log = logging.getLogger(__name__)

_SCAN_CHUNK: int = 128 * 1024 * 1024   # 128 MiB scan stride for backup hunting


def hunt_superblocks(path: str) -> list[dict[str, Any]]:
    """
    Scan *path* for primary and backup ext4 superblocks.

    Returns a list of dicts with keys: offset, score, block_size,
    blocks_per_group, is_backup, uuid, volume_name.
    """
    results: list[dict[str, Any]] = []
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        log.error("Cannot stat %s: %s", path, exc)
        return results

    with open(path, "rb", buffering=0) as f:
        fd = f.fileno()

        # Primary superblock
        sb = _read_and_parse(fd, SUPERBLOCK_OFFSET, path)
        if sb and sb.is_valid:
            results.append(_sb_to_dict(sb, is_backup=False))

            # Backup superblocks
            for off in backup_superblock_offsets(
                sb.block_size, sb.blocks_per_group, sb.total_groups
            ):
                if off + 1024 > size:
                    break
                bsb = _read_and_parse(fd, off, path)
                if bsb and bsb.is_valid:
                    results.append(_sb_to_dict(bsb, is_backup=True))

    return results


def _read_and_parse(fd: int, offset: int, path: str):  # type: ignore[return]
    try:
        buf = os.pread(fd, 1024, offset)
        return parse_superblock(buf, raw_offset=offset)
    except OSError as exc:
        log.debug("I/O error reading SB at %d in %s: %s", offset, path, exc)
        return None


def _sb_to_dict(sb: Any, is_backup: bool) -> dict[str, Any]:
    return {
        "offset":         sb.raw_offset,
        "score":          sb.score,
        "block_size":     sb.block_size,
        "blocks_per_group": sb.blocks_per_group,
        "is_backup":      is_backup,
        "uuid":           sb.uuid.hex() if sb.uuid else "",
        "volume_name":    sb.volume_name,
    }
