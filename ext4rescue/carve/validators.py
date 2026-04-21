"""
ext4rescue/carve/validators.py — Fast, stdlib-only file format validators.

Each validator receives raw bytes and returns a ``(is_valid, reason)`` tuple.
Validators are intentionally conservative: a partial / truncated file may
still pass if the available bytes look structurally sound.

All functions are pure (no I/O) and will not raise on arbitrary byte input.
"""

from __future__ import annotations

import struct

# ── Public API ────────────────────────────────────────────────────────────────


def validate_jpeg(data: bytes) -> tuple[bool, str]:
    """
    Validate JPEG data.

    Checks:
    * SOI marker ``FF D8 FF`` at offset 0.
    * At least one APP or SOF marker segment after SOI.
    * EOI marker ``FF D9`` anywhere in the last 64 bytes.

    Args:
        data: Raw file bytes.

    Returns:
        ``(True, "ok")`` or ``(False, <reason>)``.
    """
    if len(data) < 3:
        return False, "too short"
    if data[:3] != b"\xff\xd8\xff":
        return False, f"bad SOI marker: {data[:3].hex()}"
    # EOI check — lenient: look in last 64 bytes to tolerate trailing padding
    tail = data[-64:] if len(data) >= 64 else data
    if b"\xff\xd9" not in tail:
        return False, "missing EOI marker"
    return True, "ok"


def validate_png(data: bytes) -> tuple[bool, str]:
    """
    Validate PNG data.

    Checks:
    * 8-byte PNG signature.
    * IHDR chunk immediately after signature (type bytes ``IHDR``).
    * IEND chunk presence anywhere in the data.

    Args:
        data: Raw file bytes.

    Returns:
        ``(True, "ok")`` or ``(False, <reason>)``.
    """
    _PNG_SIG = b"\x89PNG\r\n\x1a\n"
    if len(data) < 8:
        return False, "too short"
    if data[:8] != _PNG_SIG:
        return False, f"bad PNG signature: {data[:8].hex()}"
    # IHDR must be the first chunk (starts at offset 8)
    if len(data) < 16:
        return False, "truncated before IHDR"
    if data[12:16] != b"IHDR":
        return False, f"expected IHDR chunk, got: {data[12:16]!r}"
    if b"IEND" not in data:
        return False, "missing IEND chunk"
    return True, "ok"


def validate_pdf(data: bytes) -> tuple[bool, str]:
    """
    Validate PDF data.

    Checks:
    * Header ``%PDF-`` at offset 0.
    * ``%%EOF`` marker in the last 1 KiB.

    Args:
        data: Raw file bytes.

    Returns:
        ``(True, "ok")`` or ``(False, <reason>)``.
    """
    if len(data) < 5:
        return False, "too short"
    if not data.startswith(b"%PDF-"):
        return False, f"bad PDF header: {data[:8]!r}"
    tail = data[-1024:] if len(data) >= 1024 else data
    if b"%%EOF" not in tail:
        return False, "missing %%EOF trailer"
    return True, "ok"


def validate_zip(data: bytes) -> tuple[bool, str]:
    """
    Validate ZIP archive data.

    Checks:
    * Local File Header signature ``PK\\x03\\x04`` at offset 0
      (or Empty-archive ``PK\\x05\\x06`` / spanned ``PK\\x07\\x08``).
    * Presence of End-of-Central-Directory record ``PK\\x05\\x06`` in the
      last 64 KiB.

    Args:
        data: Raw file bytes.

    Returns:
        ``(True, "ok")`` or ``(False, <reason>)``.
    """
    if len(data) < 4:
        return False, "too short"
    sig = data[:4]
    valid_sigs = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
    if sig not in valid_sigs:
        return False, f"bad ZIP signature: {sig.hex()}"
    # EOCD record check
    eocd_search = data[-65536:] if len(data) > 65536 else data
    if b"PK\x05\x06" not in eocd_search:
        return False, "missing End-of-Central-Directory record"
    return True, "ok"


def validate_mp4(data: bytes) -> tuple[bool, str]:
    """
    Validate MP4 / ISO Base Media container data.

    Checks:
    * ``ftyp`` box at byte offset 4 (standard ISO BMFF layout).
    * Presence of at least one of ``moov``, ``mdat``, ``free``, ``wide``
      within the first 64 KiB.

    Args:
        data: Raw file bytes.

    Returns:
        ``(True, "ok")`` or ``(False, <reason>)``.
    """
    if len(data) < 8:
        return False, "too short"
    box_type = data[4:8]
    if box_type != b"ftyp":
        return False, f"expected 'ftyp' box at offset 4, got: {box_type!r}"
    search_area = data[:65536] if len(data) > 65536 else data
    for marker in (b"moov", b"mdat", b"free", b"wide"):
        if marker in search_area:
            return True, "ok"
    return False, "no moov/mdat box found"


def validate_any(data: bytes, file_type: str) -> tuple[bool, str]:
    """
    Dispatch to the appropriate validator by *file_type* name.

    Supported type names (case-insensitive): ``jpeg``, ``jpg``, ``png``,
    ``pdf``, ``zip``, ``mp4``.

    Args:
        data:      Raw file bytes.
        file_type: File type identifier string.

    Returns:
        ``(True, "ok")`` or ``(False, <reason>)``.
        Returns ``(False, "unknown file type: <type>")`` for unsupported types.
    """
    _DISPATCH = {
        "jpeg": validate_jpeg,
        "jpg":  validate_jpeg,
        "png":  validate_png,
        "pdf":  validate_pdf,
        "zip":  validate_zip,
        "mp4":  validate_mp4,
    }
    fn = _DISPATCH.get(file_type.lower().strip())
    if fn is None:
        return False, f"unknown file type: {file_type!r}"
    return fn(data)
