"""
ext4rescue/scan/fs_detector.py — Multi-filesystem detection from raw disk bytes.

Detects ext4, ZFS, NTFS, FAT32, and exFAT by inspecting magic bytes and
known on-disk structures.  Returns a list of :class:`~ext4rescue.models.FSMatch`
dicts sorted by confidence (highest first).

No writes are performed; the caller supplies pre-read byte buffers.
"""

from __future__ import annotations

import struct
from typing import Any

# ── Magic constants ───────────────────────────────────────────────────────────

EXT4_MAGIC: int = 0xEF53
"""ext4 superblock magic at offset 56 (0x38) *within* the superblock, which
is itself at byte 1080 from the start of the partition for block_size ≥ 2048."""

ZFS_UBERBLOCK_MAGIC: int = 0x00BAB10C
"""ZFS uberblock magic (big-endian u64 low word)."""

_ZFS_LABEL_PATTERNS: tuple[bytes, ...] = (
    b"\x0c\xb1\xba\x00",   # little-endian fragment of uberblock magic
    b"\x00\xba\xb1\x0c",   # big-endian
    b"ZPOOL",
    b"ZFS!",
)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_filesystem(
    data: bytes,
    tail_data: bytes | None = None,
) -> list[dict[str, Any]]:
    """
    Identify filesystem signatures in a raw byte buffer.

    Args:
        data:       Leading bytes of the disk/partition (≥ 4 KiB recommended,
                    but the function degrades gracefully with less).
        tail_data:  Trailing bytes of the disk (used for ZFS end-label
                    detection).  Pass ``None`` when unavailable.

    Returns:
        List of result dicts, each containing:

        * ``type``       — filesystem type string
        * ``offset``     — byte offset of the detected signature
        * ``confidence`` — float in ``[0.0, 1.0]``
        * ``details``    — dict with per-type metadata

        Sorted by ``confidence`` descending.  Multiple results for the same
        filesystem type are possible (e.g. ZFS labels at both ends).
    """
    results: list[dict[str, Any]] = []

    results.extend(_detect_ext4(data))
    results.extend(_detect_zfs(data, tail_data))
    results.extend(_detect_ntfs(data))
    results.extend(_detect_fat32(data))
    results.extend(_detect_exfat(data))

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results


# Backward-compatible alias used by the legacy scan path
def detect_filesystems(path: str) -> list[dict[str, Any]]:
    """
    Detect filesystems by reading the first and last 4 KiB of *path*.

    This is a thin convenience wrapper around :func:`detect_filesystem` that
    opens the file itself.  Prefer calling :func:`detect_filesystem` directly
    when you already hold the bytes in memory.

    Args:
        path: Path to a disk image or block device (opened read-only).

    Returns:
        Same format as :func:`detect_filesystem`.
    """
    import os
    head = b""
    tail = b""
    try:
        size = os.path.getsize(path)
        with open(path, "rb", buffering=0) as f:
            head = f.read(4096)
            if size > 4096:
                f.seek(max(0, size - 4096))
                tail = f.read(4096)
    except OSError:
        pass
    return detect_filesystem(head, tail or None)


# ── Per-filesystem detectors ──────────────────────────────────────────────────

def _detect_ext4(data: bytes) -> list[dict[str, Any]]:
    """Detect ext4 via the superblock magic at byte 1080."""
    results: list[dict[str, Any]] = []
    if len(data) < 1082:
        return results

    magic = struct.unpack_from("<H", data, 1080)[0]
    if magic != EXT4_MAGIC:
        return results

    details: dict[str, Any] = {"magic": hex(magic)}

    # Opportunistically read block size and state from the superblock
    if len(data) >= 1024 + 0x3C:
        try:
            log_block_size = struct.unpack_from("<I", data, 1024 + 0x18)[0]
            if log_block_size <= 6:
                details["block_size"] = 1024 << log_block_size
            state = struct.unpack_from("<H", data, 1024 + 0x3A)[0]
            details["state"] = hex(state)
        except struct.error:
            pass

    results.append({
        "type": "ext4",
        "offset": 1024,
        "confidence": 0.95,
        "details": details,
    })
    return results


def _detect_zfs(
    data: bytes, tail_data: bytes | None
) -> list[dict[str, Any]]:
    """
    Detect ZFS labels.

    ZFS stores two labels at the beginning (L0 at 0, L1 at 256 KiB) and two
    at the end (L2, L3) of the vdev.  We check the provided head/tail buffers
    for known patterns.
    """
    results: list[dict[str, Any]] = []

    def _scan(buf: bytes, base_offset: int, label: str) -> None:
        for pattern in _ZFS_LABEL_PATTERNS:
            idx = buf.find(pattern)
            if idx != -1:
                results.append({
                    "type": "zfs",
                    "offset": base_offset + idx,
                    "confidence": 0.80,
                    "details": {
                        "label": label,
                        "pattern": pattern.hex(),
                        "position": "start" if base_offset == 0 else "end",
                    },
                })
                break  # one result per buffer side

    if data:
        _scan(data, 0, "L0/L1")
    if tail_data:
        _scan(tail_data, 0, "L2/L3")

    return results


def _detect_ntfs(data: bytes) -> list[dict[str, Any]]:
    """Detect NTFS via the OEM ID at bytes 3–10."""
    if len(data) < 11:
        return []
    if data[3:11] != b"NTFS    ":
        return []
    return [{
        "type": "ntfs",
        "offset": 0,
        "confidence": 0.90,
        "details": {"oem_id": "NTFS    "},
    }]


def _detect_fat32(data: bytes) -> list[dict[str, Any]]:
    """Detect FAT32 via boot signature 0x55AA and FS type string."""
    if len(data) < 512:
        return []
    boot_sig = struct.unpack_from("<H", data, 510)[0] if len(data) >= 512 else 0
    if boot_sig != 0x55AA:
        return []
    fs_type = data[82:90] if len(data) >= 90 else b""
    if b"FAT32" not in fs_type:
        return []
    return [{
        "type": "fat32",
        "offset": 0,
        "confidence": 0.85,
        "details": {"fs_type": fs_type.decode("ascii", errors="replace").strip()},
    }]


def _detect_exfat(data: bytes) -> list[dict[str, Any]]:
    """Detect exFAT via the OEM ID 'EXFAT   ' at bytes 3–10."""
    if len(data) < 11:
        return []
    if data[3:11] != b"EXFAT   ":
        return []
    return [{
        "type": "exfat",
        "offset": 0,
        "confidence": 0.88,
        "details": {"oem_id": "EXFAT   "},
    }]
