import os

CHUNK_SIZE = 64 * 1024 * 1024


def read_at(fd: int, offset: int, size: int) -> bytes:
    return os.pread(fd, size, offset)


def scan_chunks(path: str, chunk_size: int = CHUNK_SIZE):
    size = os.path.getsize(path)
    with open(path, "rb", buffering=0) as f:
        fd = f.fileno()
        offset = 0
        while offset < size:
            data = read_at(fd, offset, min(chunk_size, size - offset))
            yield offset, data
            offset += chunk_size
