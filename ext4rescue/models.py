"""
ext4rescue/models.py — Shared dataclasses used across the ext4rescue pipeline.

All dataclasses expose ``to_dict()`` / ``from_dict()`` helpers for plain JSON
serialisation without introducing external dependencies.

Requires Python 3.11+ (used as the project baseline); all fields carry type
annotations and ``from __future__ import annotations`` is used for forward
references.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


# ── SuperblockCandidate ───────────────────────────────────────────────────────

@dataclass
class SuperblockCandidate:
    """A candidate ext4 superblock location found during a scan."""

    offset: int = 0
    """Byte offset of the superblock on disk."""

    score: int = 0
    """Validity score in [0, 100]; ≥ 50 is considered usable."""

    block_size: int = 4096
    """Block size in bytes derived from s_log_block_size."""

    blocks_per_group: int = 0
    inodes_per_group: int = 0
    uuid: str = ""
    volume_name: str = ""
    is_backup: bool = False
    """True when this candidate came from a backup group, not group 0."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SuperblockCandidate":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── FSMatch ───────────────────────────────────────────────────────────────────

@dataclass
class FSMatch:
    """Result of a filesystem type detection pass."""

    fs_type: str = ""
    """Detected filesystem type, e.g. ``'ext4'``, ``'zfs'``, ``'ntfs'``."""

    offset: int = 0
    """Byte offset from the start of the image where the FS was detected."""

    confidence: float = 0.0
    """Detection confidence in [0.0, 1.0]."""

    details: dict[str, Any] = field(default_factory=dict)
    """Extra per-type information (magic bytes, label, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FSMatch":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── RecoveryConfig ────────────────────────────────────────────────────────────

@dataclass
class RecoveryConfig:
    """Runtime configuration for a recovery session."""

    disk_path: str = ""
    """Path to the disk image or block device (read-only)."""

    output_dir: str = "recovery_out"
    """Directory where recovered files will be written."""

    block_size: int = 4096
    min_file_size: int = 512
    """Skip recovered files smaller than this (bytes)."""

    max_file_size: int = 8 * 1024 * 1024 * 1024   # 8 GiB
    carve_fallback: bool = True
    """Enable file carving when filesystem-aware recovery fails."""

    journal_scan: bool = True
    """Scan the ext4 journal for deleted filename candidates."""

    skip_zeroed_blocks: bool = True
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RecoveryConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── FileEntry ─────────────────────────────────────────────────────────────────

@dataclass
class FileEntry:
    """Represents a single recovered or candidate file."""

    inode_nr: int = 0
    path: str = ""
    """Reconstructed path (relative to FS root).  Empty when unknown."""

    name: str = ""
    """Filename component only; may differ from path basename for orphans."""

    file_type: int = 0
    """ext4 file type constant (1=regular, 2=dir, 5=symlink, etc.)."""

    size_bytes: int = 0
    mtime: int = 0
    """Modification time as a Unix timestamp (seconds)."""

    atime: int = 0
    ctime: int = 0
    recovered_path: str = ""
    """Absolute path on the *host* filesystem where the file was saved."""

    is_orphan: bool = False
    confidence: float = 1.0
    """Recovery confidence: 1.0 = name from directory entry; lower = guessed."""

    source: str = "fs"
    """How the entry was found: ``'fs'``, ``'journal'``, ``'carve'``."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FileEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── ScanRegion ────────────────────────────────────────────────────────────────

@dataclass
class ScanRegion:
    """A contiguous region of the disk image assigned to a category."""

    start: int = 0
    end: int = 0
    """Exclusive end byte offset."""

    region_type: str = "unknown"
    """One of: ``'zfs_label'``, ``'ext4_metadata'``, ``'data'``,
    ``'zeroed'``, ``'unknown'``."""

    confidence: float = 0.0

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScanRegion":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── ProgressState ─────────────────────────────────────────────────────────────

@dataclass
class ProgressState:
    """Snapshot of the recovery pipeline's current progress."""

    phase: str = "idle"
    """Current phase name, e.g. ``'scan'``, ``'recover'``, ``'carve'``."""

    total_bytes: int = 0
    processed_bytes: int = 0
    files_recovered: int = 0
    files_carving: int = 0
    errors: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0

    @property
    def progress_pct(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return min(100.0, self.processed_bytes / self.total_bytes * 100.0)

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at if self.finished_at else time.time()
        return end - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProgressState":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── JournalNameCandidate ──────────────────────────────────────────────────────

@dataclass
class JournalNameCandidate:
    """A filename recovered from a jbd2 journal transaction."""

    name: str = ""
    """Filename as found in the journaled directory block."""

    inode_nr: int = 0
    parent_inode_nr: int = 0
    file_type: int = 0

    transaction_seq: int = 0
    """jbd2 transaction sequence number (big-endian on disk, stored as int)."""

    commit_time: int = 0
    """Unix timestamp from the jbd2 commit block, if available; else 0."""

    confidence: float = 0.5
    """Lower than a live directory entry; higher when corroborated by inode."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JournalNameCandidate":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
