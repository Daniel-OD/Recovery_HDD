from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .validators import validate_any


SIGNATURES = {
    b"\xFF\xD8\xFF": ("jpg", b"\xFF\xD9"),
    b"\x89PNG\r\n\x1a\n": ("png", b"IEND"),
    b"%PDF-": ("pdf", b"%%EOF"),
    b"PK\x03\x04": ("zip", b"PK\x05\x06"),
    b"ftyp": ("mp4", None),
}


@dataclass
class CarvingResult:
    carved_files: list[str] = field(default_factory=list)
    error_count: int = 0
    output_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "carved_files": self.carved_files,
            "error_count": self.error_count,
            "output_dir": self.output_dir,
        }


def run_carving(disk_path: str, output_dir: str) -> CarvingResult:
    os.makedirs(output_dir, exist_ok=True)
    result = CarvingResult(output_dir=output_dir)

    with open(disk_path, "rb") as f:
        offset = 0
        chunk_size = 4 * 1024 * 1024
        overlap = 64 * 1024

        while True:
            data = f.read(chunk_size)
            if not data:
                break

            for sig, (ext, end_marker) in SIGNATURES.items():
                idx = 0
                while True:
                    pos = data.find(sig, idx)
                    if pos == -1:
                        break
                    abs_pos = offset + pos

                    try:
                        f.seek(abs_pos)
                        blob = f.read(64 * 1024 * 1024)
                        if end_marker:
                            end = blob.find(end_marker)
                            if end != -1:
                                blob = blob[: end + len(end_marker)]
                        valid, _ = validate_any(blob, ext)
                        if valid:
                            out_path = os.path.join(output_dir, f"carved_{abs_pos}.{ext}")
                            with open(out_path, "wb") as out:
                                out.write(blob)
                            result.carved_files.append(out_path)
                    except Exception:
                        result.error_count += 1

                    idx = pos + 1

            offset += chunk_size - overlap
            f.seek(offset)

    return result
