"""
ext4/gdt.py — Group Descriptor Table (GDT) parser.

Fiecare block group din ext4 are un descriptor care indică unde se află
pe disc bitmap-urile, inode table-ul și alte structuri ale grupului.

Structuri:
    ext4_group_desc (32 bytes) — layout vechi, fără 64bit
    ext4_group_desc (64 bytes) — layout nou, cu 64bit feature
"""

import struct
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ── Dimensiuni descriptor ─────────────────────────────────────────────────────
GDT_ENTRY_SIZE_32  = 32   # ext4 fără INCOMPAT_64BIT
GDT_ENTRY_SIZE_64  = 64   # ext4 cu  INCOMPAT_64BIT (desc_size=64)

# ── Format struct grup descriptor 32-byte ─────────────────────────────────────
# Referință: Linux fs/ext4/ext4.h — struct ext4_group_desc
_GDT32_FORMAT = "<"  \
    "I"   \
    "I"   \
    "I"   \
    "H"   \
    "H"   \
    "H"   \
    "H"   \
    "H"   \
    "H"   \
    "I"

_GDT32_FIELDS = [
    "bg_block_bitmap_lo",      # 0x00  Block bitmap location (low 32)
    "bg_inode_bitmap_lo",      # 0x04  Inode bitmap location (low 32)
    "bg_inode_table_lo",       # 0x08  Inode table location (low 32)
    "bg_free_blocks_count_lo", # 0x0C  Free blocks count (low 16)
    "bg_free_inodes_count_lo", # 0x0E  Free inodes count (low 16)
    "bg_used_dirs_count_lo",   # 0x10  Used directories count (low 16)
    "bg_flags",                # 0x12  Group flags (INODE_UNINIT, etc.)
    "bg_exclude_bitmap_lo",    # 0x14  Snapshot exclusion bitmap (low 32) - 2 x H
    "bg_block_bitmap_csum_lo", # 0x18  Block bitmap checksum (low 16)
    "bg_inode_bitmap_csum_lo", # 0x1A  Inode bitmap checksum (low 16)
    # Restul până la 32 bytes
]

# Format simplu pentru parsing rapid
_GDT32_PACK = "<IIIHHHHHHI"

# ── Flags grup ────────────────────────────────────────────────────────────────
BG_INODE_UNINIT    = 0x0001   # Inode table/bitmap neinițializate
BG_BLOCK_UNINIT   = 0x0002   # Block bitmap neinițializat
BG_INODE_ZEROED   = 0x0004   # Inode table zeroed


@dataclass
class GroupDescriptor:
    """Descriptor pentru un block group ext4."""

    group_nr: int = 0

    # Locații pe disc (low 32 bits; high 32 bits pentru 64bit FS)
    block_bitmap_lo: int  = 0
    inode_bitmap_lo: int  = 0
    inode_table_lo: int   = 0

    # High 32 bits (pentru FS > 2^32 blocuri, cu INCOMPAT_64BIT)
    block_bitmap_hi: int  = 0
    inode_bitmap_hi: int  = 0
    inode_table_hi: int   = 0

    # Contoare
    free_blocks_count: int = 0
    free_inodes_count: int = 0
    used_dirs_count: int   = 0

    # Flags
    flags: int = 0

    # Validitate
    is_valid: bool = False

    # ── Proprietăți ──────────────────────────────────────────────────────────

    @property
    def block_bitmap(self) -> int:
        """Numărul de bloc al block bitmap-ului (combinat 48/64bit)."""
        return self.block_bitmap_lo | (self.block_bitmap_hi << 32)

    @property
    def inode_bitmap(self) -> int:
        """Numărul de bloc al inode bitmap-ului."""
        return self.inode_bitmap_lo | (self.inode_bitmap_hi << 32)

    @property
    def inode_table(self) -> int:
        """Numărul primului bloc al inode table-ului."""
        return self.inode_table_lo | (self.inode_table_hi << 32)

    @property
    def inode_uninit(self) -> bool:
        return bool(self.flags & BG_INODE_UNINIT)

    @property
    def block_uninit(self) -> bool:
        return bool(self.flags & BG_BLOCK_UNINIT)

    def inode_offset(self, inode_nr: int, inodes_per_group: int,
                     inode_size: int, block_size: int) -> int:
        """
        Calculează offset-ul absolut pe disc al unui inode specific.

        Formula:
            bg  = (inode_nr - 1) // inodes_per_group
            idx = (inode_nr - 1) % inodes_per_group
            offset = inode_table_block * block_size + idx * inode_size
        """
        idx = (inode_nr - 1) % inodes_per_group
        table_byte_offset = self.inode_table * block_size
        return table_byte_offset + idx * inode_size


# ── Parser GDT ────────────────────────────────────────────────────────────────

def parse_group_descriptor(buf: bytes, group_nr: int,
                            desc_size: int = 32,
                            is_64bit: bool = False) -> GroupDescriptor:
    """
    Parsează un singur grup descriptor din bytes raw.

    Args:
        buf:       buffer de cel puțin `desc_size` bytes
        group_nr:  numărul grupului (pentru logging)
        desc_size: 32 sau 64 bytes
        is_64bit:  True dacă FS are INCOMPAT_64BIT

    Returns:
        GroupDescriptor populat
    """
    if len(buf) < 32:
        gd = GroupDescriptor(group_nr=group_nr)
        return gd

    # Câmpuri prezente în ambele variante (primii 32 bytes)
    try:
        (block_bitmap_lo, inode_bitmap_lo, inode_table_lo,
         free_blocks_lo, free_inodes_lo, used_dirs_lo,
         flags, _excl, _bbcsum, _ibcsum, _unused
         ) = struct.unpack_from("<IIIHHHHxxxxHHxx", buf, 0)
    except struct.error:
        # Fallback la parsing minimal
        try:
            block_bitmap_lo = struct.unpack_from("<I", buf, 0)[0]
            inode_bitmap_lo = struct.unpack_from("<I", buf, 4)[0]
            inode_table_lo  = struct.unpack_from("<I", buf, 8)[0]
            free_blocks_lo  = struct.unpack_from("<H", buf, 12)[0]
            free_inodes_lo  = struct.unpack_from("<H", buf, 14)[0]
            used_dirs_lo    = struct.unpack_from("<H", buf, 16)[0]
            flags           = struct.unpack_from("<H", buf, 18)[0]
        except struct.error:
            return GroupDescriptor(group_nr=group_nr)

    gd = GroupDescriptor(
        group_nr          = group_nr,
        block_bitmap_lo   = block_bitmap_lo,
        inode_bitmap_lo   = inode_bitmap_lo,
        inode_table_lo    = inode_table_lo,
        free_blocks_count = free_blocks_lo,
        free_inodes_count = free_inodes_lo,
        used_dirs_count   = used_dirs_lo,
        flags             = flags,
    )

    # Câmpuri 64bit (bytes 32-63, prezente dacă desc_size == 64)
    if is_64bit and desc_size >= 64 and len(buf) >= 64:
        try:
            block_bitmap_hi = struct.unpack_from("<I", buf, 0x20)[0]
            inode_bitmap_hi = struct.unpack_from("<I", buf, 0x24)[0]
            inode_table_hi  = struct.unpack_from("<I", buf, 0x28)[0]
            free_blocks_hi  = struct.unpack_from("<H", buf, 0x2C)[0]
            free_inodes_hi  = struct.unpack_from("<H", buf, 0x2E)[0]
            used_dirs_hi    = struct.unpack_from("<H", buf, 0x30)[0]

            gd.block_bitmap_hi    = block_bitmap_hi
            gd.inode_bitmap_hi    = inode_bitmap_hi
            gd.inode_table_hi     = inode_table_hi
            gd.free_blocks_count |= (free_blocks_hi << 16)
            gd.free_inodes_count |= (free_inodes_hi << 16)
            gd.used_dirs_count   |= (used_dirs_hi   << 16)
        except struct.error:
            pass

    gd.is_valid = _validate_gd(gd)
    return gd


def _validate_gd(gd: GroupDescriptor) -> bool:
    """Validare de bază a unui grup descriptor."""
    # Locațiile nu pot fi 0 (bloc 0 e rezervat) sau valori enorme
    MAX_BLOCK = 0xFFFF_FFFF_FFFF  # 48-bit max
    if gd.inode_table_lo == 0:
        return False
    if gd.inode_table > MAX_BLOCK:
        return False
    if gd.block_bitmap_lo == 0 or gd.inode_bitmap_lo == 0:
        return False
    return True


def read_gdt(disk, sb) -> list[GroupDescriptor]:
    """
    Citește întregul Group Descriptor Table de pe disc.

    GDT-ul se află imediat după superblock, în blocul următor
    față de superblock (sau la bloc 1 dacă block_size=1024,
    bloc 2 dacă block_size>=2048, deoarece SB ocupă 1 bloc).

    Args:
        disk: DiskImage deschis
        sb:   Ext4Superblock valid

    Returns:
        Listă de GroupDescriptor, unul per block group
    """
    bs         = sb.block_size
    desc_size  = max(sb.desc_size, 32)  # minim 32
    is_64bit   = sb.has_64bit
    total_grps = sb.total_groups

    # GDT-ul e în primul bloc după superblock
    # Dacă block_size=1024: SB e la bloc 1 → GDT la bloc 2
    # Dacă block_size>=2048: SB e la bloc 0 (offset 1024 în bloc) → GDT la bloc 1
    if bs == 1024:
        gdt_block = sb.raw_offset // bs + 2
    else:
        gdt_block = sb.raw_offset // bs + 1

    gdt_offset = gdt_block * bs

    log.debug(
        f"GDT: block={gdt_block}, offset={gdt_offset:,}, "
        f"groups={total_grps}, desc_size={desc_size}"
    )

    descriptors = []
    for g in range(total_grps):
        entry_offset = gdt_offset + g * desc_size
        buf = disk.read_at(entry_offset, desc_size)

        if len(buf) < 32:
            log.debug(f"  Grup {g}: buffer scurt ({len(buf)}) la offset {entry_offset:,}")
            descriptors.append(GroupDescriptor(group_nr=g))
            continue

        gd = parse_group_descriptor(buf, g, desc_size, is_64bit)
        descriptors.append(gd)

        if gd.is_valid:
            log.debug(
                f"  Grup {g:4d}: inode_table=bloc {gd.inode_table}, "
                f"free_blocks={gd.free_blocks_count}, "
                f"free_inodes={gd.free_inodes_count}"
                + (" [UNINIT]" if gd.inode_uninit else "")
            )

    valid = sum(1 for d in descriptors if d.is_valid)
    log.info(f"GDT: {valid}/{total_grps} descriptori valizi")

    return descriptors


def find_group_for_inode(inode_nr: int, inodes_per_group: int) -> tuple[int, int]:
    """
    Calculează în ce block group se află un inode.

    Returns:
        (group_nr, local_index) — indexul local în inode table
    """
    group_nr    = (inode_nr - 1) // inodes_per_group
    local_index = (inode_nr - 1) %  inodes_per_group
    return group_nr, local_index


def inode_physical_offset(inode_nr: int, gdt: list[GroupDescriptor],
                           sb) -> Optional[int]:
    """
    Calculează offset-ul fizic pe disc al unui inode.

    Args:
        inode_nr: numărul inodului (1-based, ca în ext4)
        gdt:      lista de GroupDescriptor
        sb:       Ext4Superblock

    Returns:
        Offset în bytes sau None dacă grupul e invalid
    """
    if inode_nr < 1:
        return None

    group_nr, local_idx = find_group_for_inode(inode_nr, sb.inodes_per_group)

    if group_nr >= len(gdt):
        return None

    gd = gdt[group_nr]
    if not gd.is_valid:
        return None

    return gd.inode_table * sb.block_size + local_idx * sb.inode_size
