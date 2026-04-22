"""
ext4/super.py — Ext4 superblock parsing, scoring, and backup location computation.

On-disk layout reference: Linux fs/ext4/ext4.h  ``struct ext4_super_block``.

The primary superblock is always at byte offset 1024 from the start of the
filesystem image.  Backup superblocks live at the start of certain block
groups determined by the *sparse_super* rule (groups 1 and powers of 3, 5, 7).

All on-disk integers are little-endian (LE).
"""

import math
import struct
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ── Public constants ──────────────────────────────────────────────────────────

EXT4_SUPER_MAGIC: int = 0xEF53
"""Magic number stored at offset 0x38 inside every ext4 superblock."""

SUPERBLOCK_OFFSET: int = 1024
"""Byte offset of the primary superblock from the start of the filesystem."""

SUPERBLOCK_SIZE: int = 1024
"""On-disk size of an ext4 superblock."""

MAX_SCORE: int = 100
"""Maximum possible score returned by :func:`score_superblock`."""

VALID_THRESHOLD: int = 50
"""Minimum score for a superblock to be considered usable."""

# ── Internal look-up tables ───────────────────────────────────────────────────

# Supported ext4 block sizes (bytes)
_VALID_BLOCK_SIZES: frozenset[int] = frozenset({1024, 2048, 4096, 8192})

# Supported ext4 inode sizes (bytes)
_VALID_INODE_SIZES: frozenset[int] = frozenset({128, 256, 512, 1024})

# Feature flags inside s_feature_incompat
_INCOMPAT_64BIT: int = 0x0080
_INCOMPAT_EXTENTS: int = 0x0040

# Filesystem state values
_FS_STATE_CLEAN: int = 0x0001
_FS_STATE_ERROR: int = 0x0002

# Revision levels
_REV_ORIGINAL: int = 0   # original (static) format
_REV_DYNAMIC: int = 1    # dynamic (flexible) format


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class Ext4Superblock:
    """
    Parsed representation of an ext4 on-disk superblock.

    Fields map directly to ``struct ext4_super_block`` from the Linux kernel.
    Derived properties (``block_size``, ``total_groups``, …) are computed
    on-demand from the raw fields so they always stay in sync.
    """

    raw_offset: int = 0
    """Byte offset on disk/image where this superblock was read from."""

    # ── Core fields (present in all revisions) ────────────────────────────────
    magic: int = 0
    inodes_count: int = 0
    blocks_count_lo: int = 0
    blocks_count_hi: int = 0          # 64-bit hi word; 0 for 32-bit filesystems
    r_blocks_count_lo: int = 0
    free_blocks_count_lo: int = 0
    free_inodes_count: int = 0
    first_data_block: int = 0
    log_block_size: int = 0           # block_size = 1024 << log_block_size
    blocks_per_group: int = 0
    inodes_per_group: int = 0
    state: int = 0
    creator_os: int = 0
    rev_level: int = 0

    # ── Dynamic fields (rev_level >= 1) ───────────────────────────────────────
    inode_size: int = 128
    block_group_nr: int = 0
    feature_compat: int = 0
    feature_incompat: int = 0
    feature_ro_compat: int = 0
    uuid: bytes = field(default_factory=bytes)
    volume_name: str = ""
    desc_size: int = 32               # group descriptor size (32 or 64)

    # ── Validity summary ──────────────────────────────────────────────────────
    is_valid: bool = False
    score: int = 0

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def block_size(self) -> int:
        """Block size in bytes: ``1024 << s_log_block_size``."""
        return 1024 << self.log_block_size

    @property
    def has_64bit(self) -> bool:
        """True when the filesystem uses 64-bit block addresses (INCOMPAT_64BIT)."""
        return bool(self.feature_incompat & _INCOMPAT_64BIT)

    @property
    def blocks_count(self) -> int:
        """Full 64-bit block count (lo | hi << 32)."""
        return self.blocks_count_lo | (self.blocks_count_hi << 32)

    @property
    def total_groups(self) -> int:
        """Number of block groups computed from total blocks and blocks-per-group."""
        if self.blocks_per_group == 0:
            return 0
        return math.ceil(self.blocks_count / self.blocks_per_group)


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_superblock(buf: bytes, raw_offset: int = SUPERBLOCK_OFFSET) -> Ext4Superblock:
    """
    Parse an ext4 superblock from a raw bytes buffer.

    The first byte of *buf* must correspond to the first byte of the
    superblock on disk (i.e. the caller must already have seeked to
    *raw_offset* before reading).

    Args:
        buf:        Raw bytes.  Must be at least 84 bytes to extract the magic
                    and basic geometry.  A complete superblock is 1024 bytes.
        raw_offset: Disk byte offset where this buffer was read from.  Stored
                    verbatim in :attr:`Ext4Superblock.raw_offset`.

    Returns:
        :class:`Ext4Superblock` with all parseable fields populated.
        ``is_valid`` is ``True`` only when the magic matches and the score
        meets :data:`VALID_THRESHOLD`.
    """
    sb = Ext4Superblock(raw_offset=raw_offset)

    # Need at least up to s_magic (0x38) + 2 bytes = 58 bytes
    if len(buf) < 58:
        log.warning(
            "Superblock buffer too short at offset %d: %d bytes", raw_offset, len(buf)
        )
        return sb

    # ── Bytes 0x00 – 0x37: base header ───────────────────────────────────────
    try:
        (sb.inodes_count,
         sb.blocks_count_lo,
         sb.r_blocks_count_lo,
         sb.free_blocks_count_lo,
         sb.free_inodes_count,
         sb.first_data_block,
         sb.log_block_size,
         ) = struct.unpack_from("<IIIIIII", buf, 0x00)

        sb.blocks_per_group = struct.unpack_from("<I", buf, 0x20)[0]
        sb.inodes_per_group = struct.unpack_from("<I", buf, 0x28)[0]
    except struct.error as exc:
        log.warning("Failed to unpack base header at %d: %s", raw_offset, exc)
        return sb

    # ── Bytes 0x38 – 0x4F: magic + state ─────────────────────────────────────
    try:
        sb.magic = struct.unpack_from("<H", buf, 0x38)[0]
        sb.state = struct.unpack_from("<H", buf, 0x3A)[0]
        sb.creator_os = struct.unpack_from("<I", buf, 0x48)[0]
        sb.rev_level = struct.unpack_from("<I", buf, 0x4C)[0]
    except struct.error as exc:
        log.warning("Failed to unpack state/magic at %d: %s", raw_offset, exc)
        return sb

    # ── Dynamic fields (rev_level >= 1, bytes 0x54 – 0x77) ───────────────────
    if sb.rev_level >= _REV_DYNAMIC and len(buf) >= 0x78:
        try:
            sb.inode_size = struct.unpack_from("<H", buf, 0x58)[0]
            sb.block_group_nr = struct.unpack_from("<H", buf, 0x5A)[0]
            sb.feature_compat = struct.unpack_from("<I", buf, 0x5C)[0]
            sb.feature_incompat = struct.unpack_from("<I", buf, 0x60)[0]
            sb.feature_ro_compat = struct.unpack_from("<I", buf, 0x64)[0]
            sb.uuid = bytes(buf[0x68:0x78])
        except struct.error as exc:
            log.debug("Partial dynamic fields at %d: %s", raw_offset, exc)

    # ── Volume name (bytes 0x78 – 0x87, null-terminated ASCII) ───────────────
    if len(buf) >= 0x88:
        try:
            raw_name = buf[0x78:0x88]
            sb.volume_name = raw_name.split(b"\x00")[0].decode("ascii", errors="replace")
        except Exception:
            pass  # non-fatal; leave volume_name as ""

    # ── Group descriptor size at 0xFA (250) ───────────────────────────────────
    if len(buf) >= 0xFC:
        try:
            raw_desc = struct.unpack_from("<H", buf, 0xFA)[0]
            if raw_desc in {32, 64}:
                sb.desc_size = raw_desc
        except struct.error:
            pass

    # ── 64-bit block count hi word at 0x150 (336) ────────────────────────────
    if sb.has_64bit and len(buf) >= 0x154:
        try:
            sb.blocks_count_hi = struct.unpack_from("<I", buf, 0x150)[0]
        except struct.error:
            pass

    sb.score = score_superblock(sb)
    sb.is_valid = sb.score >= VALID_THRESHOLD
    return sb


# ── Scorer ────────────────────────────────────────────────────────────────────

def score_superblock(sb: Ext4Superblock) -> int:
    """
    Rate an :class:`Ext4Superblock` on internal consistency.

    Returns a score in ``[0, MAX_SCORE]``.  Returns **0** immediately when
    the magic does not match :data:`EXT4_SUPER_MAGIC`.  A score ≥
    :data:`VALID_THRESHOLD` (50) indicates a plausibly valid superblock.

    Scoring breakdown (totals to 100 for a perfect entry):

    * **+30** correct magic
    * **+15** block size in {1024, 2048, 4096, 8192}
    * **+10** blocks_per_group > 0 and ≤ 0x10000
    * **+10** inodes_per_group > 0
    * **+10** inode_size in {128, 256, 512, 1024}
    * **+5**  rev_level in {0, 1}
    * **+5**  inodes_count > 0
    * **+5**  blocks_count > 0
    * **+5**  state in {FS_CLEAN, FS_ERROR}
    * **+5**  first_data_block consistent with block_size
    """
    if sb.magic != EXT4_SUPER_MAGIC:
        return 0

    score = 30  # magic is correct

    if sb.block_size in _VALID_BLOCK_SIZES:
        score += 15
    else:
        # Invalid block size is a strong indicator of a corrupt/absent superblock;
        # return early so the total stays well below VALID_THRESHOLD.
        return score

    if 0 < sb.blocks_per_group <= 0x10000:
        score += 10

    if sb.inodes_per_group > 0:
        score += 10

    if sb.inode_size in _VALID_INODE_SIZES:
        score += 10

    if sb.rev_level in {_REV_ORIGINAL, _REV_DYNAMIC}:
        score += 5

    if sb.inodes_count > 0:
        score += 5

    if sb.blocks_count_lo > 0:
        score += 5

    if sb.state in {_FS_STATE_CLEAN, _FS_STATE_ERROR}:
        score += 5

    # first_data_block: 0 for block_size > 1024; 1 for block_size == 1024
    expected_fdb = 0 if sb.block_size > 1024 else 1
    if sb.first_data_block == expected_fdb:
        score += 5

    return min(score, MAX_SCORE)


# ── Backup-superblock offset computation ──────────────────────────────────────

def _is_sparse_group(group: int) -> bool:
    """
    Return ``True`` when *group* should hold a superblock backup under the
    *sparse_super* feature rule.

    Groups 0 and 1 always hold a copy.  For groups > 1 the rule is: odd
    numbers that are a perfect power of 3, 5, or 7.

    Reference: ``ext4_group_sparse()`` in ``fs/ext4/super.c``.
    """
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


def backup_superblock_offsets(
    block_size: int,
    blocks_per_group: int,
    total_groups: int,
) -> list[int]:
    """
    Compute byte offsets of candidate backup superblocks on disk.

    For each backup group *g* > 0 the backup superblock is located at the
    very start of the group's first block::

        offset = g × blocks_per_group × block_size

    (Unlike the primary superblock which is at byte 1024 inside block 0,
    backup copies in groups 1+ occupy the entire first block of their group.)

    Args:
        block_size:       Filesystem block size in bytes.
        blocks_per_group: Number of blocks per group.
        total_groups:     Total number of block groups in the filesystem.

    Returns:
        Sorted list of byte offsets where backup superblocks may exist.
        Group 0 (primary) is excluded.
    """
    offsets: list[int] = []
    for g in range(1, total_groups):
        if _is_sparse_group(g):
            offsets.append(g * blocks_per_group * block_size)
    return offsets
