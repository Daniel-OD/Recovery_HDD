"""
ext4rescue/scan/region_map.py — Disk region classification map.

A :class:`RegionMap` records what type of data occupies each byte range of a
disk image.  Regions are non-overlapping; adding a new region that overlaps an
existing one replaces or trims the existing entry.

Region types
------------
* ``'zfs_label'``      — ZFS pool label area (start / end of vdev)
* ``'ext4_metadata'``  — ext4 superblock, GDT, bitmaps, inode tables
* ``'data'``           — regular file data blocks
* ``'zeroed'``         — all-zero blocks (likely wiped / unwritten)
* ``'unknown'``        — not yet classified
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field, asdict
from typing import Any

# ── Region confidence thresholds ──────────────────────────────────────────────
_SAFE_CONFIDENCE: float = 0.70
"""Regions with confidence ≥ this value are considered 'safe' to read."""

_DAMAGED_CONFIDENCE: float = 0.30
"""Regions with confidence < this value are flagged as 'damaged'."""

# Colour palette for the HTML visualisation (type → CSS hex colour)
_TYPE_COLOURS: dict[str, str] = {
    "zfs_label":     "#e74c3c",   # red
    "ext4_metadata": "#3498db",   # blue
    "data":          "#2ecc71",   # green
    "zeroed":        "#95a5a6",   # grey
    "unknown":       "#f39c12",   # orange
}

_MERGEABLE_TYPES: frozenset[str] = frozenset({
    "data", "zeroed", "unknown",
})
"""Region types that may be merged with adjacent identical-type neighbours."""


@dataclass(order=True)
class _Region:
    """Internal representation of a classified disk region."""

    start: int
    end: int           # exclusive
    region_type: str
    confidence: float

    def __len__(self) -> int:
        return max(0, self.end - self.start)


class RegionMap:
    """
    Non-overlapping classification map of a disk image.

    Regions are stored in sorted order by start offset.  Adding a new region
    that overlaps existing entries splits or removes the overlapping parts so
    that no byte is classified twice.  The most recently added region always
    wins.

    Example::

        rmap = RegionMap()
        rmap.add_region(0, 512 * 1024, "zfs_label", 0.95)
        rmap.add_region(1024, 2048, "ext4_metadata", 0.99)
        safe = rmap.safe_regions()
    """

    def __init__(self) -> None:
        self._regions: list[_Region] = []

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_region(
        self,
        start: int,
        end: int,
        region_type: str,
        confidence: float,
    ) -> None:
        """
        Classify the byte range ``[start, end)`` as *region_type*.

        Any existing regions that overlap ``[start, end)`` are trimmed or
        removed.  Adjacent regions of the same type (and compatible confidence)
        are merged when *region_type* is in the mergeable set.

        Args:
            start:       Inclusive start byte offset.
            end:         Exclusive end byte offset.
            region_type: One of the recognised type strings.
            confidence:  Detection confidence in ``[0.0, 1.0]``.

        Raises:
            ValueError: When ``start >= end`` or ``confidence`` is not in
                        ``[0.0, 1.0]``.
        """
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end})")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        new_region = _Region(start=start, end=end,
                             region_type=region_type, confidence=confidence)
        updated: list[_Region] = []

        for existing in self._regions:
            if existing.end <= new_region.start or existing.start >= new_region.end:
                # No overlap — keep as-is
                updated.append(existing)
            else:
                # Trim the overlapping portion
                if existing.start < new_region.start:
                    updated.append(_Region(
                        existing.start, new_region.start,
                        existing.region_type, existing.confidence,
                    ))
                if existing.end > new_region.end:
                    updated.append(_Region(
                        new_region.end, existing.end,
                        existing.region_type, existing.confidence,
                    ))

        updated.append(new_region)
        updated.sort(key=lambda r: r.start)
        self._regions = self._merge_adjacent(updated)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_region(self, offset: int) -> dict[str, Any] | None:
        """
        Return the region containing *offset*, or ``None`` if unclassified.

        Args:
            offset: Byte offset to look up.

        Returns:
            Dict with keys ``start``, ``end``, ``region_type``, ``confidence``
            or ``None``.
        """
        for r in self._regions:
            if r.start <= offset < r.end:
                return asdict(r)
        return None

    def safe_regions(self) -> list[tuple[int, int]]:
        """
        Return ``(start, end)`` pairs for regions with high confidence.

        Only regions whose confidence ≥ :data:`_SAFE_CONFIDENCE` and whose
        type is not ``'zfs_label'`` are returned (ZFS labels should not be
        treated as recoverable ext4 data even when confidently identified).
        """
        return [
            (r.start, r.end)
            for r in self._regions
            if r.confidence >= _SAFE_CONFIDENCE and r.region_type != "zfs_label"
        ]

    def damaged_regions(self) -> list[tuple[int, int]]:
        """
        Return ``(start, end)`` pairs for regions with low confidence.

        Regions with confidence < :data:`_DAMAGED_CONFIDENCE` are considered
        damaged or unclassifiable.
        """
        return [
            (r.start, r.end)
            for r in self._regions
            if r.confidence < _DAMAGED_CONFIDENCE
        ]

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of all regions."""
        return {
            "regions": [asdict(r) for r in self._regions],
            "total_classified_bytes": sum(len(r) for r in self._regions),
        }

    # ── HTML visualisation ────────────────────────────────────────────────────

    def to_html_visualization(self, disk_size: int) -> str:
        """
        Render a simple horizontal bar chart of the region map as HTML.

        The chart is a single ``<div>`` containing colored segments scaled
        proportionally to *disk_size*.  A legend is appended below.

        Args:
            disk_size: Total disk size in bytes (used to compute proportions).

        Returns:
            A self-contained HTML fragment (no ``<html>`` / ``<body>`` wrapper).
        """
        if disk_size <= 0:
            return "<p>No disk size available for visualisation.</p>"

        bar_width_px = 800
        bar_height_px = 40
        segments: list[str] = []

        for r in self._regions:
            left_pct = r.start / disk_size * 100
            width_pct = max(0.05, (r.end - r.start) / disk_size * 100)
            colour = _TYPE_COLOURS.get(r.region_type, "#bdc3c7")
            tip = (
                f"{r.region_type} [{r.start:,}–{r.end:,}] "
                f"conf={r.confidence:.2f}"
            )
            segments.append(
                f'<div style="position:absolute;left:{left_pct:.4f}%;'
                f'width:{width_pct:.4f}%;height:100%;'
                f'background:{colour};opacity:0.85;" '
                f'title="{html.escape(tip)}"></div>'
            )

        bar_html = (
            f'<div style="position:relative;width:{bar_width_px}px;'
            f'height:{bar_height_px}px;background:#ecf0f1;'
            f'border:1px solid #bdc3c7;overflow:hidden;">'
            + "".join(segments)
            + "</div>"
        )

        # Legend
        legend_items = "".join(
            f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{colour};margin-right:4px;vertical-align:middle;'
            f'border:1px solid #999;"></span>'
            f'<span style="margin-right:16px;">{html.escape(rtype)}</span>'
            for rtype, colour in _TYPE_COLOURS.items()
        )
        legend_html = f'<div style="margin-top:6px;font-size:12px;">{legend_items}</div>'

        return f"<div>{bar_html}{legend_html}</div>"

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _merge_adjacent(regions: list[_Region]) -> list[_Region]:
        """Merge consecutive regions of the same mergeable type."""
        if not regions:
            return regions
        merged: list[_Region] = [regions[0]]
        for curr in regions[1:]:
            prev = merged[-1]
            if (
                prev.end == curr.start
                and prev.region_type == curr.region_type
                and prev.region_type in _MERGEABLE_TYPES
                and abs(prev.confidence - curr.confidence) < 0.05
            ):
                merged[-1] = _Region(
                    prev.start, curr.end,
                    prev.region_type,
                    (prev.confidence + curr.confidence) / 2,
                )
            else:
                merged.append(curr)
        return merged
