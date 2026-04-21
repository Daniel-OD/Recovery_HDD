import os
import struct

EXT4_MAGIC = 0xEF53


def detect_filesystems(path: str):
    matches = []
    with open(path, "rb", buffering=0) as f:
        data = f.read(4096)

        # ext4
        if len(data) > 1080:
            magic = struct.unpack_from("<H", data, 1080)[0]
            if magic == EXT4_MAGIC:
                matches.append({"fs": "ext4", "offset": 1024})

        # NTFS
        if data[3:11] == b"NTFS    ":
            matches.append({"fs": "ntfs", "offset": 0})

        # FAT32
        if data[82:90] == b"FAT32   ":
            matches.append({"fs": "fat32", "offset": 0})

    size = os.path.getsize(path)
    with open(path, "rb", buffering=0) as f:
        f.seek(max(0, size - 1024))
        tail = f.read(1024)
        if b"ZFS" in tail:
            matches.append({"fs": "zfs", "offset": size - 1024})

    return matches
