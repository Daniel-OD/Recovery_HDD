"""
ext4rescue/cli.py — Command-line interface.

Commands
--------
detect-fs      Detect filesystem type(s) from the first/last bytes of the image.
hunt-super     Scan the image for ext4 superblock candidates.
recover-named  Filesystem-aware recovery: exports named files and orphans.
carve          Fallback file carving pipeline.
report         Generate an HTML report from a completed recovery session.
ai-orphans     Generate AI suggestions for orphan organization from JSON input.
ai-journal     Generate AI interpretation for journal-derived candidate names.
ai-report      Generate AI executive summary from recovery report JSON.

All commands are read-only; the source disk/image is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import MISSING, asdict, fields
from typing import Any

# ── Optional rich output (graceful fallback to plain text) ────────────────────
try:
    from rich.console import Console as _Console
    from rich.table import Table as _Table
    _console = _Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    _console = None  # type: ignore[assignment]


def _print(msg: str) -> None:
    if _HAS_RICH and _console is not None:
        _console.print(msg)
    else:
        print(msg)


def _print_err(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)


AI_MAX_ITEMS = 300


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="ext4rescue",
        description="ext4 data recovery from overwritten disks (read-only).",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    # detect-fs
    p_detect = sub.add_parser("detect-fs", help="Detect filesystem type(s).")
    p_detect.add_argument("path", help="Disk image or block device.")

    # hunt-super
    p_hunt = sub.add_parser("hunt-super", help="Find ext4 superblock candidates.")
    p_hunt.add_argument("path", help="Disk image or block device.")
    p_hunt.add_argument("--json", action="store_true", help="Output as JSON.")

    # recover-named
    p_recover = sub.add_parser(
        "recover-named",
        help="Filesystem-aware recovery (named files + orphans).",
    )
    p_recover.add_argument("path", help="Disk image or block device.")
    p_recover.add_argument(
        "--output", "-o", default="recovery_out",
        help="Output directory (default: recovery_out).",
    )

    # carve
    p_carve = sub.add_parser("carve", help="Fallback file carving.")
    p_carve.add_argument("path", help="Disk image or block device.")
    p_carve.add_argument(
        "--output", "-o", default="carve_out",
        help="Output directory (default: carve_out).",
    )

    # report
    p_report = sub.add_parser("report", help="Generate an HTML report.")
    p_report.add_argument("path", help="Disk image or block device (for display).")
    p_report.add_argument(
        "--output", "-o", default="report.html",
        help="Output HTML file (default: report.html).",
    )

    # ai-orphans
    p_ai = sub.add_parser(
        "ai-orphans",
        help="AI suggestions for orphan structure (JSON in/out).",
    )
    p_ai.add_argument("input_json", help="Input JSON containing an 'orphans' list.")
    p_ai.add_argument("output_json", help="Output JSON with AI suggestions.")
    p_ai.add_argument("--model", default="gpt-5", help="OpenAI model name.")
    p_ai.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Confidence threshold for accepted suggestions (default: 0.80).",
    )

    # ai-journal
    p_ai_journal = sub.add_parser(
        "ai-journal",
        help="AI interpretation of journal candidates (JSON in/out).",
    )
    p_ai_journal.add_argument(
        "input_json",
        help="Input JSON containing an 'items' list.",
    )
    p_ai_journal.add_argument(
        "output_json",
        help="Output JSON with AI journal interpretation.",
    )
    p_ai_journal.add_argument("--model", default="gpt-5", help="OpenAI model name.")
    p_ai_journal.add_argument(
        "--debug",
        action="store_true",
        help="Print raw AI response before writing output JSON.",
    )

    # ai-report
    p_ai_report = sub.add_parser(
        "ai-report",
        help="AI summary generation from report JSON (JSON in/out).",
    )
    p_ai_report.add_argument("input_json", help="Input report JSON.")
    p_ai_report.add_argument("output_json", help="Output JSON with AI summary.")
    p_ai_report.add_argument("--model", default="gpt-5", help="OpenAI model name.")
    p_ai_report.add_argument(
        "--debug",
        action="store_true",
        help="Print raw AI response before writing output JSON.",
    )

    args = parser.parse_args()

    if args.cmd is None:
        parser.print_help()
        return

    try:
        _dispatch(args)
    except KeyboardInterrupt:
        _print_err("Interrupted by user.")
        sys.exit(1)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def _dispatch(args: argparse.Namespace) -> None:
    """Route to the correct command handler."""
    if args.cmd == "detect-fs":
        _cmd_detect_fs(args)
    elif args.cmd == "hunt-super":
        _cmd_hunt_super(args)
    elif args.cmd == "recover-named":
        _cmd_recover_named(args)
    elif args.cmd == "carve":
        _cmd_carve(args)
    elif args.cmd == "report":
        _cmd_report(args)
    elif args.cmd == "ai-orphans":
        _cmd_ai_orphans(args)
    elif args.cmd == "ai-journal":
        _cmd_ai_journal(args)
    elif args.cmd == "ai-report":
        _cmd_ai_report(args)


# ── detect-fs ─────────────────────────────────────────────────────────────────

def _cmd_detect_fs(args: argparse.Namespace) -> None:
    from .scan.fs_detector import detect_filesystems
    try:
        results = detect_filesystems(args.path)
    except FileNotFoundError:
        _print_err(f"File not found: {args.path}")
        sys.exit(1)
    except PermissionError:
        _print_err(f"Permission denied: {args.path}")
        sys.exit(1)
    except ValueError as exc:
        _print_err(f"Failed to detect filesystem: {exc}")
        sys.exit(1)

    if not results:
        _print("No known filesystem signatures detected.")
        return

    for r in results:
        fs   = r.get("type", "?")
        off  = r.get("offset", 0)
        conf = r.get("confidence", 0.0)
        _print(f"  {fs:10s}  offset={off:,}  confidence={conf:.2f}")


# ── hunt-super ────────────────────────────────────────────────────────────────

def _cmd_hunt_super(args: argparse.Namespace) -> None:
    from .scan.super_hunter import hunt_superblocks
    try:
        results = hunt_superblocks(args.path)
    except FileNotFoundError:
        _print_err(f"File not found: {args.path}")
        sys.exit(1)
    except PermissionError:
        _print_err(f"Permission denied: {args.path}")
        sys.exit(1)

    if getattr(args, "json", False):
        print(json.dumps(results, indent=2))
        return

    if not results:
        _print("No valid superblocks found.")
        return

    for r in results:
        backup = " [backup]" if r.get("is_backup") else " [primary]"
        _print(
            f"  offset={r['offset']:,}  score={r['score']}  "
            f"block_size={r['block_size']}{backup}  "
            f"vol={r['volume_name']!r}"
        )

    _print(f"\nTotal: {len(results)} superblock(s) found.")


# ── recover-named ─────────────────────────────────────────────────────────────

def _cmd_recover_named(args: argparse.Namespace) -> None:
    from .ext4.recover import run_recovery
    try:
        os.makedirs(args.output, exist_ok=True)
        result = run_recovery(disk_path=args.path, output_dir=args.output)
    except FileNotFoundError:
        _print_err(f"File not found: {args.path}")
        sys.exit(1)
    except PermissionError:
        _print_err(f"Permission denied: {args.path}")
        sys.exit(1)
    except ValueError as exc:
        _print_err(f"Invalid argument: {exc}")
        sys.exit(1)

    for w in result.warnings:
        _print(f"  [warn] {w}")

    _print(
        f"\nRecovery summary:\n"
        f"  Named files : {result.named_count}\n"
        f"  Orphans     : {result.orphan_count}\n"
        f"  Errors      : {result.error_count}\n"
        f"  Output dir  : {result.output_dir}"
    )


# ── carve ─────────────────────────────────────────────────────────────────────

def _cmd_carve(args: argparse.Namespace) -> None:
    from .carve.engine import run_carving
    try:
        os.makedirs(args.output, exist_ok=True)
        result = run_carving(disk_path=args.path, output_dir=args.output)
    except FileNotFoundError:
        _print_err(f"File not found: {args.path}")
        sys.exit(1)
    except PermissionError:
        _print_err(f"Permission denied: {args.path}")
        sys.exit(1)
    except ValueError as exc:
        _print_err(f"Invalid argument: {exc}")
        sys.exit(1)

    for w in result.warnings:
        _print(f"  [warn] {w}")

    _print(
        f"\nCarving summary:\n"
        f"  Carved  : {result.carved_count}\n"
        f"  Skipped : {result.skipped_count}\n"
        f"  Errors  : {result.error_count}\n"
        f"  Output  : {result.output_dir}"
    )


# ── report ────────────────────────────────────────────────────────────────────

def _cmd_report(args: argparse.Namespace) -> None:
    from .report.html import generate_report
    try:
        out = generate_report(
            output_path=args.output,
            disk_path=args.path,
        )
    except FileNotFoundError:
        _print_err(f"File not found: {args.path}")
        sys.exit(1)
    except PermissionError:
        _print_err(f"Permission denied writing report: {args.output}")
        sys.exit(1)
    except ValueError as exc:
        _print_err(f"Report error: {exc}")
        sys.exit(1)

    _print(f"Report written to: {out}")


# ── ai-orphans ──────────────────────────────────────────────────────────────────

def _cmd_ai_orphans(args: argparse.Namespace) -> None:
    from .ai import OrphanRecord, OrphanRebuilder

    try:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        _print_err(f"File not found: {args.input_json}")
        sys.exit(1)
    except PermissionError:
        _print_err(f"Permission denied: {args.input_json}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        _print_err(f"Invalid JSON in input file: {exc}")
        sys.exit(1)

    try:
        raw_orphans = data["orphans"]
        orphans = [OrphanRecord(**item) for item in raw_orphans]
    except (KeyError, TypeError, ValueError) as exc:
        _print_err(f"Invalid orphan input format: {exc}")
        sys.exit(1)

    try:
        rebuilder = OrphanRebuilder(
            model=args.model,
            confidence_threshold=args.threshold,
        )
        result = rebuilder.suggest_structure(orphans)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    except PermissionError:
        _print_err(f"Permission denied writing output: {args.output_json}")
        sys.exit(1)
    except ValueError as exc:
        _print_err(f"AI orphan suggestion error: {exc}")
        sys.exit(1)

    _print(f"AI orphan suggestions written to: {args.output_json}")


def _cmd_ai_journal(args: argparse.Namespace) -> None:
    from .ai import JournalCandidate, JournalInterpreter

    try:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        _print_err(f"File not found: {args.input_json}")
        sys.exit(1)
    except PermissionError:
        _print_err(f"Permission denied: {args.input_json}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        _print_err(f"Invalid JSON in input file: {exc}")
        sys.exit(1)

    try:
        _validate_input_object(data, "journal input")
        raw_items = _validate_input_items(
            data,
            key="items",
            label="journal items",
            max_items=AI_MAX_ITEMS,
        )
        items = _map_dataclass_list(raw_items, JournalCandidate, "journal item")
    except ValueError as exc:
        _print_err(f"Invalid journal input format: {exc}")
        sys.exit(1)

    try:
        interpreter = JournalInterpreter(model=args.model)
        result = interpreter.interpret(items, max_items=AI_MAX_ITEMS)
        payload = {
            "events": [asdict(event) for event in result.events],
            "notes": list(result.notes),
        }
        _validate_journal_output_payload(payload)
        if getattr(args, "debug", False):
            _print(f"Raw AI response:\n{result.raw_response}")
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except PermissionError:
        _print_err(f"Permission denied writing output: {args.output_json}")
        sys.exit(1)
    except RuntimeError as exc:
        _print_err(str(exc))
        sys.exit(1)
    except ValueError as exc:
        _print_err(f"AI journal interpretation error: {exc}")
        sys.exit(1)

    _print(f"AI journal interpretation written to: {args.output_json}")


def _cmd_ai_report(args: argparse.Namespace) -> None:
    from .ai import ReportAI

    try:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        _print_err(f"File not found: {args.input_json}")
        sys.exit(1)
    except PermissionError:
        _print_err(f"Permission denied: {args.input_json}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        _print_err(f"Invalid JSON in input file: {exc}")
        sys.exit(1)

    try:
        _validate_input_object(data, "report input")
        _validate_report_input(data, max_items=AI_MAX_ITEMS)
        summarizer = ReportAI(model=args.model)
        result = summarizer.summarize(data, max_items=AI_MAX_ITEMS)
        payload = {
            "executive_summary": result.executive_summary,
            "highlights": list(result.highlights),
            "warnings": list(result.warnings),
            "next_steps": list(result.next_steps),
            "operator_notes": list(result.operator_notes),
        }
        _validate_report_output_payload(payload)
        if getattr(args, "debug", False):
            _print(f"Raw AI response:\n{result.raw_response}")
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except PermissionError:
        _print_err(f"Permission denied writing output: {args.output_json}")
        sys.exit(1)
    except RuntimeError as exc:
        _print_err(str(exc))
        sys.exit(1)
    except ValueError as exc:
        _print_err(f"AI report summary error: {exc}")
        sys.exit(1)

    _print(f"AI report summary written to: {args.output_json}")


def _validate_input_object(data: Any, label: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a top-level JSON object.")


def _validate_input_items(
    data: dict[str, Any],
    *,
    key: str,
    label: str,
    max_items: int,
) -> list[Any]:
    if key not in data:
        raise ValueError(f"Missing required field '{key}'.")
    raw_items = data[key]
    if not isinstance(raw_items, list):
        raise ValueError(f"Field '{key}' must be a JSON array.")
    if len(raw_items) > max_items:
        raise ValueError(f"{label} count {len(raw_items)} exceeds max_items={max_items}.")
    return raw_items


def _map_dataclass_list(
    raw_items: list[Any],
    cls: type[Any],
    label: str,
) -> list[Any]:
    mapped: list[Any] = []
    cls_fields = {f.name: f for f in fields(cls)}
    required_fields = {
        name
        for name, f in cls_fields.items()
        if f.default is MISSING and f.default_factory is MISSING
    }
    allowed_fields = set(cls_fields.keys())

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{idx}] must be an object.")
        missing = sorted(required_fields - set(item.keys()))
        extra = sorted(set(item.keys()) - allowed_fields)
        if missing or extra:
            chunks: list[str] = []
            if missing:
                chunks.append(f"missing required field(s): {', '.join(missing)}")
            if extra:
                chunks.append(f"unexpected field(s): {', '.join(extra)}")
            raise ValueError(f"{label}[{idx}] invalid mapping ({'; '.join(chunks)}).")
        try:
            mapped.append(cls(**item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{idx}] could not be mapped: {exc}") from exc
    return mapped


def _validate_report_input(data: dict[str, Any], *, max_items: int) -> None:
    for key, value in data.items():
        if isinstance(value, list) and len(value) > max_items:
            raise ValueError(f"Input list '{key}' has {len(value)} items (max_items={max_items}).")


def _validate_journal_output_payload(payload: dict[str, Any]) -> None:
    if set(payload.keys()) != {"events", "notes"}:
        raise ValueError("Journal output must contain exactly: events, notes.")
    if not isinstance(payload["events"], list):
        raise ValueError("Journal output field 'events' must be an array.")
    if not isinstance(payload["notes"], list):
        raise ValueError("Journal output field 'notes' must be an array.")
    for idx, event in enumerate(payload["events"]):
        if not isinstance(event, dict):
            raise ValueError(f"Journal output events[{idx}] must be an object.")
        if set(event.keys()) != {"inode_nr", "candidate_name", "confidence"}:
            raise ValueError(
                f"Journal output events[{idx}] must contain exactly: inode_nr, candidate_name, confidence."
            )
        if not isinstance(event["inode_nr"], int):
            raise ValueError(f"Journal output events[{idx}].inode_nr must be an integer.")
        if not isinstance(event["candidate_name"], str):
            raise ValueError(f"Journal output events[{idx}].candidate_name must be a string.")
        if not isinstance(event["confidence"], (int, float)):
            raise ValueError(f"Journal output events[{idx}].confidence must be a number.")
    for idx, note in enumerate(payload["notes"]):
        if not isinstance(note, str):
            raise ValueError(f"Journal output notes[{idx}] must be a string.")


def _validate_report_output_payload(payload: dict[str, Any]) -> None:
    expected = {"executive_summary", "highlights", "warnings", "next_steps", "operator_notes"}
    if set(payload.keys()) != expected:
        raise ValueError(
            "Report output must contain exactly: executive_summary, highlights, warnings, next_steps, operator_notes."
        )
    if not isinstance(payload["executive_summary"], str):
        raise ValueError("Report output field 'executive_summary' must be a string.")
    for key in ("highlights", "warnings", "next_steps", "operator_notes"):
        value = payload[key]
        if not isinstance(value, list):
            raise ValueError(f"Report output field '{key}' must be an array.")
        for idx, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"Report output {key}[{idx}] must be a string.")
