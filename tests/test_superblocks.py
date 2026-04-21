"""
tests/test_superblocks.py — pytest test suite for ext4rescue.ext4.super.

Coverage:
- parse_superblock detects EXT4_SUPER_MAGIC 0xEF53
- score_superblock returns MAX_SCORE for a perfect synthetic superblock
- candidate backup superblock offsets are correct for
  block_size=4096, blocks_per_group=32768
- group 1 backup offset equals 134 217 728
- wrong magic yields score 0
- invalid block size yields score below VALID_THRESHOLD
"""

import struct

import pytest

from ext4rescue.ext4.super import (
    EXT4_SUPER_MAGIC,
    MAX_SCORE,
    VALID_THRESHOLD,
    Ext4Superblock,
    backup_superblock_offsets,
    parse_superblock,
    score_superblock,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_superblock_bytes(
    *,
    magic: int = EXT4_SUPER_MAGIC,
    log_block_size: int = 2,          # → 4096-byte blocks
    blocks_per_group: int = 32768,
    inodes_per_group: int = 8192,
    inodes_count: int = 1_000_000,
    blocks_count_lo: int = 10_000_000,
    rev_level: int = 1,
    inode_size: int = 256,
    state: int = 0x0001,              # FS_STATE_CLEAN
    first_data_block: int = 0,        # correct for block_size > 1024
) -> bytes:
    """
    Build a minimal synthetic ext4 superblock buffer (1024 bytes).

    All fields not explicitly specified are zero.  The returned bytes can be
    passed directly to :func:`parse_superblock`.
    """
    buf = bytearray(1024)

    struct.pack_into("<I", buf, 0x00, inodes_count)        # s_inodes_count
    struct.pack_into("<I", buf, 0x04, blocks_count_lo)     # s_blocks_count_lo
    struct.pack_into("<I", buf, 0x14, first_data_block)    # s_first_data_block
    struct.pack_into("<I", buf, 0x18, log_block_size)      # s_log_block_size
    struct.pack_into("<I", buf, 0x20, blocks_per_group)    # s_blocks_per_group
    struct.pack_into("<I", buf, 0x28, inodes_per_group)    # s_inodes_per_group
    struct.pack_into("<H", buf, 0x38, magic)               # s_magic
    struct.pack_into("<H", buf, 0x3A, state)               # s_state
    struct.pack_into("<I", buf, 0x4C, rev_level)           # s_rev_level
    struct.pack_into("<H", buf, 0x58, inode_size)          # s_inode_size (dyn)

    return bytes(buf)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_sb_bytes() -> bytes:
    """1024-byte buffer for a perfect synthetic ext4 superblock."""
    return build_superblock_bytes()


@pytest.fixture
def perfect_sb(perfect_sb_bytes: bytes) -> Ext4Superblock:
    """Parsed :class:`Ext4Superblock` from the perfect synthetic buffer."""
    return parse_superblock(perfect_sb_bytes)


# ── parse_superblock ───────────────────────────────────────────────────────────

class TestParseSuperblock:
    """Tests for :func:`parse_superblock`."""

    def test_correct_magic_detected(self, perfect_sb: Ext4Superblock) -> None:
        assert perfect_sb.magic == EXT4_SUPER_MAGIC

    def test_is_valid_for_perfect_superblock(self, perfect_sb: Ext4Superblock) -> None:
        assert perfect_sb.is_valid is True

    def test_block_size_derived_from_log_block_size(self, perfect_sb: Ext4Superblock) -> None:
        # log_block_size = 2  →  1024 << 2 = 4096
        assert perfect_sb.block_size == 4096

    def test_wrong_magic_is_invalid(self) -> None:
        buf = build_superblock_bytes(magic=0xDEAD)
        sb = parse_superblock(buf)
        assert sb.is_valid is False

    def test_zero_magic_is_invalid(self) -> None:
        buf = build_superblock_bytes(magic=0x0000)
        sb = parse_superblock(buf)
        assert sb.is_valid is False

    def test_buffer_too_short_returns_empty_superblock(self) -> None:
        sb = parse_superblock(b"\x00" * 10)
        assert sb.magic == 0
        assert sb.is_valid is False

    def test_raw_offset_stored(self) -> None:
        buf = build_superblock_bytes()
        sb = parse_superblock(buf, raw_offset=134_217_728)
        assert sb.raw_offset == 134_217_728

    def test_inodes_count_parsed(self, perfect_sb: Ext4Superblock) -> None:
        assert perfect_sb.inodes_count == 1_000_000

    def test_blocks_per_group_parsed(self, perfect_sb: Ext4Superblock) -> None:
        assert perfect_sb.blocks_per_group == 32768

    def test_inode_size_parsed_dynamic(self, perfect_sb: Ext4Superblock) -> None:
        assert perfect_sb.inode_size == 256

    def test_volume_name_parsed(self) -> None:
        buf = bytearray(build_superblock_bytes())
        buf[0x78:0x80] = b"testvol\x00"
        sb = parse_superblock(bytes(buf))
        assert sb.volume_name == "testvol"

    @pytest.mark.parametrize("log_bs, expected_bs", [
        (0, 1024),
        (1, 2048),
        (2, 4096),
        (3, 8192),
    ])
    def test_block_size_computed_for_all_valid_log_values(
        self, log_bs: int, expected_bs: int
    ) -> None:
        first_data = 1 if expected_bs == 1024 else 0
        buf = build_superblock_bytes(log_block_size=log_bs, first_data_block=first_data)
        sb = parse_superblock(buf)
        assert sb.block_size == expected_bs


# ── score_superblock ───────────────────────────────────────────────────────────

class TestScoreSuperblock:
    """Tests for :func:`score_superblock`."""

    def test_perfect_superblock_scores_maximum(self, perfect_sb: Ext4Superblock) -> None:
        assert score_superblock(perfect_sb) == MAX_SCORE

    def test_perfect_superblock_score_equals_100(self, perfect_sb: Ext4Superblock) -> None:
        assert MAX_SCORE == 100
        assert score_superblock(perfect_sb) == 100

    def test_wrong_magic_scores_zero(self) -> None:
        buf = build_superblock_bytes(magic=0x1234)
        sb = parse_superblock(buf)
        assert score_superblock(sb) == 0

    def test_zero_magic_scores_zero(self) -> None:
        buf = build_superblock_bytes(magic=0x0000)
        sb = parse_superblock(buf)
        assert score_superblock(sb) == 0

    def test_score_never_negative(self) -> None:
        buf = build_superblock_bytes(magic=0xFFFF)
        sb = parse_superblock(buf)
        assert score_superblock(sb) >= 0

    @pytest.mark.parametrize("bad_log", [7, 15, 20, 31])
    def test_invalid_block_size_scores_below_threshold(self, bad_log: int) -> None:
        """Enormous log_block_size values produce sizes outside the valid set."""
        buf = build_superblock_bytes(log_block_size=bad_log)
        sb = parse_superblock(buf)
        assert score_superblock(sb) < VALID_THRESHOLD

    @pytest.mark.parametrize("log_bs, fdb", [
        (0, 1),   # 1024-byte blocks → first_data_block must be 1
        (1, 0),
        (2, 0),
        (3, 0),
    ])
    def test_valid_block_sizes_score_above_threshold(
        self, log_bs: int, fdb: int
    ) -> None:
        buf = build_superblock_bytes(log_block_size=log_bs, first_data_block=fdb)
        sb = parse_superblock(buf)
        assert score_superblock(sb) >= VALID_THRESHOLD

    def test_score_capped_at_max(self, perfect_sb: Ext4Superblock) -> None:
        assert score_superblock(perfect_sb) <= MAX_SCORE


# ── backup_superblock_offsets ──────────────────────────────────────────────────

class TestBackupSuperblockOffsets:
    """Tests for :func:`backup_superblock_offsets`."""

    # The anchor assertion required by the problem statement
    def test_group1_offset_equals_134217728(self) -> None:
        """Group 1 backup: 1 × 32768 × 4096 = 134 217 728."""
        offsets = backup_superblock_offsets(
            block_size=4096, blocks_per_group=32768, total_groups=10
        )
        assert 134_217_728 in offsets

    def test_group1_only_when_total_groups_is_two(self) -> None:
        offsets = backup_superblock_offsets(
            block_size=4096, blocks_per_group=32768, total_groups=2
        )
        assert offsets == [134_217_728]

    def test_group0_not_in_backup_list(self) -> None:
        """Primary superblock (group 0) must not appear as a backup."""
        offsets = backup_superblock_offsets(
            block_size=4096, blocks_per_group=32768, total_groups=10
        )
        assert 0 not in offsets
        assert 1024 not in offsets   # primary SB byte offset

    def test_group3_offset_correct(self) -> None:
        # 3 × 32768 × 4096 = 402 653 184
        offsets = backup_superblock_offsets(
            block_size=4096, blocks_per_group=32768, total_groups=10
        )
        assert 3 * 32768 * 4096 in offsets

    def test_even_group2_not_in_sparse_backups(self) -> None:
        """Group 2 is even and > 1, so sparse_super skips it."""
        offsets = backup_superblock_offsets(
            block_size=4096, blocks_per_group=32768, total_groups=10
        )
        assert 2 * 32768 * 4096 not in offsets

    def test_offsets_are_sorted(self) -> None:
        offsets = backup_superblock_offsets(
            block_size=4096, blocks_per_group=32768, total_groups=30
        )
        assert offsets == sorted(offsets)

    def test_total_groups_zero_returns_empty(self) -> None:
        assert backup_superblock_offsets(4096, 32768, 0) == []

    def test_total_groups_one_returns_empty(self) -> None:
        """Only group 0 exists — no backup groups."""
        assert backup_superblock_offsets(4096, 32768, 1) == []

    @pytest.mark.parametrize("group, in_sparse", [
        (1, True),    # always in sparse
        (3, True),    # 3^1
        (5, True),    # 5^1
        (7, True),    # 7^1
        (9, True),    # 3^2
        (25, True),   # 5^2
        (27, True),   # 3^3
        (49, True),   # 7^2
        (2,  False),  # even
        (4,  False),  # even
        (6,  False),  # even
        (10, False),  # even
        (11, False),  # odd but not a power of 3/5/7
        (15, False),  # odd but 3×5, not a pure power
    ])
    def test_sparse_group_membership(self, group: int, in_sparse: bool) -> None:
        offsets = backup_superblock_offsets(
            block_size=4096, blocks_per_group=32768, total_groups=100
        )
        offset = group * 32768 * 4096
        assert (offset in offsets) is in_sparse

    @pytest.mark.parametrize("block_size, bpg", [
        (1024, 8192),
        (2048, 16384),
        (4096, 32768),
        (8192, 32768),
    ])
    def test_group1_offset_formula_for_various_geometries(
        self, block_size: int, bpg: int
    ) -> None:
        offsets = backup_superblock_offsets(
            block_size=block_size, blocks_per_group=bpg, total_groups=4
        )
        expected = 1 * bpg * block_size
        assert expected in offsets
