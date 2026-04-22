"""
ext4rescue/ext4/journal.py — jbd2 journal miner for deleted filename recovery.

This module scans the ext4 journal (jbd2) for descriptor blocks that reference
filesystem directory blocks, then attempts to parse those directory blocks as
``ext4_dir_entry_2`` structures and extract filenames.

**Endianness note**: jbd2 header fields are **big-endian**; ext4 filesystem
structures (directory entries, etc.) are **little-endian**.

This is a *miner*, not a full replay engine: we tolerate corruption, skip
invalid blocks, and never modify the source image.

Reference: Linux ``include/linux/jbd2.h``.
"""

from __future__ import annotations

import struct
import logging
from dataclasses import dataclass, field
from typing import Any

from ..models import JournalNameCandidate

log = logging.getLogger(__name__)

# ── jbd2 constants (big-endian on disk) ───────────────────────────────────────

JBD2_MAGIC: int = 0xC03B3998
"""Magic number in every jbd2 journal block header (stored BE)."""

# Block types
JBD2_DESCRIPTOR_BLOCK: int = 1
JBD2_COMMIT_BLOCK: int = 2
JBD2_SUPERBLOCK_V1: int = 3
JBD2_SUPERBLOCK_V2: int = 4
JBD2_REVOKE_BLOCK: int = 5

# journal_block_tag_s flags
JBD2_FLAG_ESCAPE: int = 0x01      # block contains escaped magic
JBD2_FLAG_SAME_UUID: int = 0x02   # no UUID follows this tag
JBD2_FLAG_DELETED: int = 0x04     # this block was deleted
JBD2_FLAG_LAST_TAG: int = 0x08    # last tag in descriptor

# Minimum supported journal block size (bytes)
_MIN_BLOCK_SIZE: int = 1024

# ext4 dir-entry minimum size
_DIR_ENTRY_MIN: int = 8

# ext4 file type values
_FILE_TYPES: dict[int, str] = {
    0: "unknown",
    1: "regular",
    2: "directory",
    3: "char_dev",
    4: "block_dev",
    5: "fifo",
    6: "socket",
    7: "symlink",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TransactionSummary:
    """Summary of one parsed jbd2 transaction."""

    sequence: int = 0
    commit_time: int = 0          # 0 if not found / not parsed
    descriptor_blocks: int = 0
    dir_blocks_scanned: int = 0
    candidates: list[JournalNameCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "commit_time": self.commit_time,
            "descriptor_blocks": self.descriptor_blocks,
            "dir_blocks_scanned": self.dir_blocks_scanned,
            "candidates": [c.to_dict() for c in self.candidates],
            "warnings": self.warnings,
        }


# ── Journal superblock parsing ────────────────────────────────────────────────

@dataclass
class JournalSuperblock:
    """Parsed jbd2 journal superblock (enough fields for mining)."""

    block_size: int = 4096
    maxlen: int = 0              # total blocks in journal
    first: int = 1               # first block of log
    sequence: int = 0            # first commit sequence expected
    start: int = 0               # block number of first log entry
    is_valid: bool = False


def parse_journal_superblock(buf: bytes) -> JournalSuperblock:
    """
    Parse a jbd2 journal superblock (version 2).

    The journal superblock is the first block of the journal inode's data.
    All fields are **big-endian**.

    Args:
        buf: Raw bytes of the journal's first block.

    Returns:
        :class:`JournalSuperblock`; ``is_valid`` is False on parse failure.
    """
    jsb = JournalSuperblock()
    if len(buf) < 0x58:
        log.debug("Journal superblock buffer too short: %d bytes", len(buf))
        return jsb

    try:
        # journal_header_s (12 bytes at offset 0)
        magic, block_type, seq = struct.unpack_from(">III", buf, 0)
        if magic != JBD2_MAGIC:
            log.debug("Bad journal SB magic: 0x%08X", magic)
            return jsb
        if block_type not in {JBD2_SUPERBLOCK_V1, JBD2_SUPERBLOCK_V2}:
            log.debug("Block type 0x%X is not a journal superblock", block_type)
            return jsb

        # journal_superblock_s fields (big-endian)
        block_size = struct.unpack_from(">I", buf, 0x0C)[0]
        maxlen = struct.unpack_from(">I", buf, 0x10)[0]
        first = struct.unpack_from(">I", buf, 0x14)[0]
        sequence = struct.unpack_from(">I", buf, 0x18)[0]
        start = struct.unpack_from(">I", buf, 0x1C)[0]

        if block_size < _MIN_BLOCK_SIZE or block_size > 65536:
            log.debug("Implausible journal block size: %d", block_size)
            return jsb

        jsb.block_size = block_size
        jsb.maxlen = maxlen
        jsb.first = first
        jsb.sequence = sequence
        jsb.start = start
        jsb.is_valid = True

    except struct.error as exc:
        log.debug("Failed to parse journal superblock: %s", exc)

    return jsb


# ── Journal scanning ──────────────────────────────────────────────────────────

def mine_journal(
    journal_data: bytes,
    journal_block_size: int = 4096,
    fs_block_size: int = 4096,
) -> list[JournalNameCandidate]:
    """
    Mine filename candidates from raw jbd2 journal data.

    Iterates over all journal blocks looking for descriptor blocks, extracts
    the filesystem block numbers they reference, then scans those data blocks
    for directory entries.

    Args:
        journal_data:       Raw bytes of the entire journal (all blocks).
        journal_block_size: Block size used by the journal (bytes).
        fs_block_size:      Filesystem block size; used only to sanity-check
                            block numbers from descriptor tags.

    Returns:
        Deduplicated list of :class:`JournalNameCandidate` sorted by
        transaction sequence number then filename.
    """
    if len(journal_data) < journal_block_size:
        log.debug("Journal data too short for mining (%d bytes)", len(journal_data))
        return []

    total_blocks = len(journal_data) // journal_block_size
    log.info("Journal mining: %d blocks × %d bytes", total_blocks, journal_block_size)

    all_candidates: list[JournalNameCandidate] = []

    # Two-pass approach:
    # Pass 1 — collect (sequence, [data_buf]) from descriptor + commit blocks
    # Pass 2 — already inline: parse data blocks immediately after each descriptor

    block_idx = 0
    while block_idx < total_blocks:
        block_off = block_idx * journal_block_size
        block_buf = journal_data[block_off: block_off + journal_block_size]

        if len(block_buf) < 12:
            block_idx += 1
            continue

        try:
            magic, block_type, seq = struct.unpack_from(">III", block_buf, 0)
        except struct.error:
            block_idx += 1
            continue

        if magic != JBD2_MAGIC:
            block_idx += 1
            continue

        if block_type == JBD2_DESCRIPTOR_BLOCK:
            candidates, consumed = _process_descriptor_block(
                journal_data, block_idx, journal_block_size, seq
            )
            all_candidates.extend(candidates)
            block_idx += 1 + consumed   # 1 = descriptor block + consumed data blocks
        elif block_type == JBD2_COMMIT_BLOCK:
            # Commit blocks don't carry filenames, but do carry a timestamp
            _note_commit_block(block_buf, seq, all_candidates)
            block_idx += 1
        else:
            block_idx += 1

    # Deduplicate by (inode, name) keeping highest confidence
    deduped = _deduplicate_candidates(all_candidates)
    deduped.sort(key=lambda c: (c.transaction_seq, c.name))
    log.info("Journal mining: %d unique candidates extracted", len(deduped))
    return deduped


# ── Descriptor block processing ───────────────────────────────────────────────

def _process_descriptor_block(
    journal_data: bytes,
    desc_block_idx: int,
    block_size: int,
    sequence: int,
) -> tuple[list[JournalNameCandidate], int]:
    """
    Parse a descriptor block and scan the data blocks it references.

    Returns ``(candidates, num_data_blocks_consumed)``.
    """
    total_blocks = len(journal_data) // block_size
    desc_buf = _read_block(journal_data, desc_block_idx, block_size)
    if not desc_buf:
        return [], 0

    candidates: list[JournalNameCandidate] = []
    data_block_idx = desc_block_idx + 1
    tag_offset = 12   # skip journal_header_s

    while tag_offset + 8 <= len(desc_buf):
        tag = _parse_block_tag(desc_buf, tag_offset)
        if tag is None:
            break

        fs_block_nr, flags, tag_size = tag
        tag_offset += tag_size

        # Read the corresponding data block
        data_buf = _read_block(journal_data, data_block_idx, block_size)
        data_block_idx += 1

        if data_buf and _looks_like_dir_block(data_buf):
            cands = _parse_dir_block(data_buf, fs_block_nr, sequence)
            candidates.extend(cands)

        if flags & JBD2_FLAG_LAST_TAG:
            break

    consumed = data_block_idx - desc_block_idx - 1
    return candidates, consumed


def _parse_block_tag(
    buf: bytes, offset: int
) -> tuple[int, int, int] | None:
    """
    Parse a journal_block_tag_s at *offset* inside a descriptor block.

    Returns ``(fs_block_nr, flags, tag_size_bytes)`` or ``None`` on error.

    We support only the 32-bit block number variant (no 64-bit incompat flag
    handling) which covers the vast majority of ext4 volumes ≤ 16 TiB.
    """
    if offset + 8 > len(buf):
        return None
    try:
        t_blocknr = struct.unpack_from(">I", buf, offset)[0]
        t_flags = struct.unpack_from(">H", buf, offset + 4)[0]
        # High 16 bits of block nr sit at offset+6 in the v2 tag; ignore for now
    except struct.error:
        return None

    # Tag size: 8 bytes base; +16 if UUID present (not SAME_UUID flag)
    tag_size = 8
    if not (t_flags & JBD2_FLAG_SAME_UUID):
        tag_size += 16   # UUID bytes follow

    return t_blocknr, t_flags, tag_size


# ── Directory block parsing ───────────────────────────────────────────────────

def _looks_like_dir_block(buf: bytes) -> bool:
    """
    Heuristic: return True when *buf* may contain ext4 directory entries.

    Checks that the first entry has a plausible inode number, rec_len, and
    name_len without running a full parse.
    """
    if len(buf) < _DIR_ENTRY_MIN:
        return False
    try:
        inode_nr = struct.unpack_from("<I", buf, 0)[0]
        rec_len = struct.unpack_from("<H", buf, 4)[0]
        name_len = struct.unpack_from("<B", buf, 6)[0]
    except struct.error:
        return False

    if inode_nr == 0 or rec_len < _DIR_ENTRY_MIN:
        return False
    if name_len == 0 or name_len > 255:
        return False
    if rec_len > len(buf):
        return False
    return True


def _parse_dir_block(
    buf: bytes,
    fs_block_nr: int,
    sequence: int,
) -> list[JournalNameCandidate]:
    """
    Parse a directory block as a stream of ``ext4_dir_entry_2`` records.

    Args:
        buf:          Raw block bytes (little-endian ext4 structure).
        fs_block_nr:  Filesystem block number (for logging only).
        sequence:     jbd2 transaction sequence number.

    Returns:
        List of :class:`JournalNameCandidate` entries found in this block.
    """
    candidates: list[JournalNameCandidate] = []
    offset = 0

    while offset + _DIR_ENTRY_MIN <= len(buf):
        try:
            inode_nr = struct.unpack_from("<I", buf, offset)[0]
            rec_len = struct.unpack_from("<H", buf, offset + 4)[0]
            name_len = struct.unpack_from("<B", buf, offset + 6)[0]
            file_type = struct.unpack_from("<B", buf, offset + 7)[0]
        except struct.error:
            break

        if rec_len < _DIR_ENTRY_MIN:
            log.debug(
                "Block %d seq %d: rec_len=%d too small at offset %d — stopping",
                fs_block_nr, sequence, rec_len, offset,
            )
            break

        if offset + rec_len > len(buf):
            log.debug(
                "Block %d seq %d: rec_len=%d exceeds buffer at offset %d",
                fs_block_nr, sequence, rec_len, offset,
            )
            break

        if inode_nr > 0 and name_len > 0:
            name_end = offset + 8 + name_len
            if name_end <= len(buf):
                raw_name = buf[offset + 8: name_end]
                try:
                    name = raw_name.decode("utf-8", errors="replace")
                except Exception:
                    name = raw_name.decode("latin-1", errors="replace")

                # Skip . and .. entries
                if name not in {".", ".."}:
                    candidates.append(JournalNameCandidate(
                        name=name,
                        inode_nr=inode_nr,
                        file_type=file_type,
                        transaction_seq=sequence,
                        confidence=0.55,
                    ))

        offset += rec_len

    return candidates


# ── Commit block ──────────────────────────────────────────────────────────────

def _note_commit_block(
    buf: bytes,
    sequence: int,
    candidates: list[JournalNameCandidate],
) -> None:
    """
    Extract commit timestamp from a commit block and back-annotate
    candidates that share the same transaction sequence.

    jbd2_commit_header has the commit time at offset 16 (big-endian u32).
    """
    if len(buf) < 20:
        return
    try:
        commit_sec = struct.unpack_from(">I", buf, 16)[0]
    except struct.error:
        return

    if commit_sec == 0:
        return

    for cand in candidates:
        if cand.transaction_seq == sequence and cand.commit_time == 0:
            cand.commit_time = commit_sec


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_block(data: bytes, idx: int, block_size: int) -> bytes | None:
    """Read block *idx* from *data*; return None if out of range."""
    start = idx * block_size
    end = start + block_size
    if end > len(data):
        return None
    return data[start:end]


def _deduplicate_candidates(
    candidates: list[JournalNameCandidate],
) -> list[JournalNameCandidate]:
    """Keep the highest-confidence entry for each (inode_nr, name) pair."""
    best: dict[tuple[int, str], JournalNameCandidate] = {}
    for c in candidates:
        key = (c.inode_nr, c.name)
        if key not in best or c.confidence > best[key].confidence:
            best[key] = c
    return list(best.values())
