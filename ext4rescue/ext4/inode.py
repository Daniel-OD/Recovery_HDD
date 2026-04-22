from __future__ import annotations

import struct
from dataclasses import dataclass

EXT4_EXTENTS_FL = 0x00080000
S_IFMT = 0xF000
S_IFREG = 0x8000
S_IFDIR = 0x4000
S_IFLNK = 0xA000


@dataclass(slots=True)
class Ext4Inode:
    inode_nr: int = 0
    mode: int = 0
    uid: int = 0
    size_lo: int = 0
    atime: int = 0
    ctime: int = 0
    mtime: int = 0
    dtime: int = 0
    gid: int = 0
    links_count: int = 0
    blocks_lo: int = 0
    flags: int = 0
    i_block: bytes = b""
    size_high: int = 0
    is_valid: bool = False

    @property
    def size(self) -> int:
        return self.size_lo | (self.size_high << 32)

    @property
    def uses_extents(self) -> bool:
        return bool(self.flags & EXT4_EXTENTS_FL)

    @property
    def file_type(self) -> int:
        return self.mode & S_IFMT

    @property
    def is_regular(self) -> bool:
        return self.file_type == S_IFREG

    @property
    def is_dir(self) -> bool:
        return self.file_type == S_IFDIR

    @property
    def is_symlink(self) -> bool:
        return self.file_type == S_IFLNK

    @property
    def direct_blocks(self) -> list[int]:
        if len(self.i_block) < 48:
            return []
        return [b for b in struct.unpack_from("<12I", self.i_block, 0) if b != 0]


def parse_inode(buf: bytes, inode_nr: int = 0) -> Ext4Inode:
    ino = Ext4Inode(inode_nr=inode_nr)
    if len(buf) < 160:
        return ino
    try:
        ino.mode = struct.unpack_from("<H", buf, 0x0)[0]
        ino.uid = struct.unpack_from("<H", buf, 0x2)[0]
        ino.size_lo = struct.unpack_from("<I", buf, 0x4)[0]
        ino.atime = struct.unpack_from("<I", buf, 0x8)[0]
        ino.ctime = struct.unpack_from("<I", buf, 0xC)[0]
        ino.mtime = struct.unpack_from("<I", buf, 0x10)[0]
        ino.dtime = struct.unpack_from("<I", buf, 0x14)[0]
        ino.gid = struct.unpack_from("<H", buf, 0x18)[0]
        ino.links_count = struct.unpack_from("<H", buf, 0x1A)[0]
        ino.blocks_lo = struct.unpack_from("<I", buf, 0x1C)[0]
        ino.flags = struct.unpack_from("<I", buf, 0x20)[0]
        ino.i_block = bytes(buf[0x28:0x64])
        ino.size_high = struct.unpack_from("<I", buf, 0x6C)[0]
    except struct.error:
        return ino
    if ino.mode == 0:
        return ino
    if ino.size > (16 * 1024**4):
        return ino
    ino.is_valid = True
    return ino
