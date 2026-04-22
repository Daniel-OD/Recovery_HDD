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

All commands are read-only; the source disk/image is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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
