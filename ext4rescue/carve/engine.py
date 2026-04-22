"""
ext4rescue/carve/engine.py — Fallback file carving pipeline.

Scans the disk image for known file signatures and extracts candidate files
when filesystem-aware recovery is unavailable or incomplete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CarvingResult:
    """Summary returned by :func:`run_carving`."""

    carved_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    output_dir: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "carved_count":  self.carved_count,
            "skipped_count": self.skipped_count,
            "error_count":   self.error_count,
            "output_dir":    self.output_dir,
            "warnings":      self.warnings,
        }


def run_carving(disk_path: str, output_dir: str, **kwargs: Any) -> CarvingResult:
    """
    Run the file carving pipeline on *disk_path*.

    Args:
        disk_path:  Path to disk image or block device (read-only).
        output_dir: Directory where carved files are written.

    Returns:
        :class:`CarvingResult` with counts and status.
    """
    log.info("run_carving: disk=%s out=%s (stub)", disk_path, output_dir)
    return CarvingResult(output_dir=output_dir,
                         warnings=["Carving pipeline not yet fully implemented."])
