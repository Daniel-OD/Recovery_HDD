"""
ext4rescue/utils.py — General-purpose utility functions.

All functions are pure (no side effects on disk).  Exceptions from malformed
input are caught internally and replaced with safe fallback return values.
"""

from __future__ import annotations

import re
import struct
import datetime
from typing import Callable, TypeVar

T = TypeVar("T")

# Characters that are invalid in filenames on common Linux filesystems
_UNSAFE_CHARS: re.Pattern[str] = re.compile(r'[/\x00]')
# Characters that are unsafe but replaceable in path components
_UNSAFE_PATH_CHARS: re.Pattern[str] = re.compile(r'[\x00-\x1f\x7f/\\:*?"<>|]')
# Reserved names on case-insensitive filesystems (kept for annotation only)
_RESERVED_NAMES: frozenset[str] = frozenset({
    ".", "..", "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})


def safe_filename(name: str) -> str:
    """
    Return a sanitised version of *name* safe for use as a filename.

    Replaces NUL bytes and forward slashes with underscores, strips leading
    dots and spaces, and falls back to ``"_unnamed"`` when the result is
    empty.

    Args:
        name: Raw filename string (may contain arbitrary bytes decoded as str).

    Returns:
        A non-empty string suitable for use as a filename component.

    Examples:
        >>> safe_filename("my/file.txt")
        'my_file.txt'
        >>> safe_filename("")
        '_unnamed'
    """
    if not isinstance(name, str):
        return "_unnamed"
    sanitised = _UNSAFE_CHARS.sub("_", name)
    sanitised = sanitised.strip(". ")   # strip leading/trailing dots and spaces
    return sanitised if sanitised else "_unnamed"


def safe_path_components(parts: list[str]) -> list[str]:
    """
    Sanitise a list of path components for safe reconstruction.

    Each component has control characters, NUL, and path separators replaced
    with underscores.  Empty components and bare dots/double-dots are replaced
    with ``"_"`` to prevent path traversal.

    Args:
        parts: List of raw path component strings (e.g. from a dir-entry walk).

    Returns:
        List of sanitised, non-empty component strings of the same length.
    """
    result: list[str] = []
    for part in parts:
        if not isinstance(part, str):
            result.append("_")
            continue
        sanitised = _UNSAFE_PATH_CHARS.sub("_", part).strip()
        if not sanitised or sanitised in {".", ".."}:
            sanitised = "_"
        result.append(sanitised)
    return result


def format_timestamp(unix_ts: int) -> str:
    """
    Convert a Unix timestamp to an ISO-8601 string in UTC.

    Args:
        unix_ts: Seconds since the Unix epoch (may be 0 or negative).

    Returns:
        UTC datetime string such as ``"2023-07-14T09:32:00Z"``.
        Returns ``"1970-01-01T00:00:00Z"`` for zero / invalid values.
    """
    try:
        dt = datetime.datetime.fromtimestamp(unix_ts, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError, ValueError):
        return "1970-01-01T00:00:00Z"


def chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    """
    Partition ``[0, total)`` into consecutive half-open chunks.

    Args:
        total:      Total number of bytes / items.
        chunk_size: Size of each chunk (must be > 0).

    Returns:
        List of ``(start, end)`` tuples where ``end = min(start+chunk_size, total)``.
        Returns an empty list when *total* ≤ 0 or *chunk_size* ≤ 0.

    Examples:
        >>> chunk_ranges(10, 3)
        [(0, 3), (3, 6), (6, 9), (9, 10)]
    """
    if total <= 0 or chunk_size <= 0:
        return []
    ranges: list[tuple[int, int]] = []
    offset = 0
    while offset < total:
        end = min(offset + chunk_size, total)
        ranges.append((offset, end))
        offset = end
    return ranges


def deduplicate_by_key(items: list[T], key_fn: Callable[[T], object]) -> list[T]:
    """
    Remove duplicates from *items* keeping the first occurrence of each key.

    Args:
        items:  Input list (not modified).
        key_fn: Function that returns a hashable key for each item.

    Returns:
        New list with duplicates removed, preserving insertion order.
    """
    seen: set[object] = set()
    result: list[T] = []
    for item in items:
        k = key_fn(item)
        if k not in seen:
            seen.add(k)
            result.append(item)
    return result


# ── File format quick-validators ──────────────────────────────────────────────

def verify_jpeg(data: bytes) -> bool:
    """
    Return ``True`` when *data* looks like a valid JPEG.

    Checks for the SOI marker (FF D8 FF) at the start and the EOI marker
    (FF D9) at or near the end.

    Args:
        data: Raw file bytes (partial reads accepted; EOI check is skipped
              when fewer than 2 bytes remain after the header).
    """
    if len(data) < 3:
        return False
    if data[:3] != b"\xff\xd8\xff":
        return False
    # EOI marker check: look in the last 64 bytes for tolerance of padding
    tail = data[-64:] if len(data) >= 64 else data
    return b"\xff\xd9" in tail


def verify_pdf(data: bytes) -> bool:
    """
    Return ``True`` when *data* looks like a valid PDF.

    Checks for the ``%PDF-`` header and the ``%%EOF`` trailer.

    Args:
        data: Raw file bytes.
    """
    if not data.startswith(b"%PDF-"):
        return False
    tail = data[-1024:] if len(data) >= 1024 else data
    return b"%%EOF" in tail


def verify_zip(data: bytes) -> bool:
    """
    Return ``True`` when *data* looks like a valid ZIP archive.

    Accepts Local File Header (PK\\x03\\x04), End-of-Central-Directory
    (PK\\x05\\x06), and spanned-archive (PK\\x07\\x08) signatures.

    Args:
        data: Raw file bytes.
    """
    if len(data) < 4:
        return False
    sig = data[:4]
    return sig in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")

