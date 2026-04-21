"""
ext4rescue/report/html.py — Self-contained HTML recovery report generator.
"""

from __future__ import annotations

import html
import datetime
from typing import Any


def generate_report(
    output_path: str,
    recovery_result: Any | None = None,
    carving_result: Any | None = None,
    disk_path: str = "",
) -> str:
    """
    Write a self-contained HTML report to *output_path*.

    Args:
        output_path:      File path where the HTML will be written.
        recovery_result:  Optional :class:`~ext4rescue.ext4.recover.RecoveryResult`.
        carving_result:   Optional :class:`~ext4rescue.carve.engine.CarvingResult`.
        disk_path:        Disk image path (for display only).

    Returns:
        The absolute path of the written file.
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[str] = []
    if recovery_result is not None:
        d = recovery_result.to_dict() if hasattr(recovery_result, "to_dict") else {}
        rows.append(_kv_row("Named files recovered", d.get("named_count", "—")))
        rows.append(_kv_row("Orphan files",          d.get("orphan_count", "—")))
        rows.append(_kv_row("Recovery errors",       d.get("error_count",  "—")))
    if carving_result is not None:
        d = carving_result.to_dict() if hasattr(carving_result, "to_dict") else {}
        rows.append(_kv_row("Carved files",    d.get("carved_count",  "—")))
        rows.append(_kv_row("Carving skipped", d.get("skipped_count", "—")))

    table = f'<table border="1" cellpadding="6">{"".join(rows)}</table>' if rows else "<p>No results.</p>"

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ext4rescue Report</title>
  <style>
    body {{ font-family: monospace; margin: 2em; }}
    table {{ border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 4px 12px; }}
  </style>
</head>
<body>
  <h1>ext4rescue Recovery Report</h1>
  <p><strong>Generated:</strong> {html.escape(now)}</p>
  <p><strong>Disk:</strong> {html.escape(disk_path or "—")}</p>
  {table}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(body)

    return output_path


def _kv_row(key: str, value: Any) -> str:
    return f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
