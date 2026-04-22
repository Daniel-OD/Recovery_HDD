#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

AI_MAX_ITEMS = 300


def _fail(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        _fail(f"File not found: {path}")
    except json.JSONDecodeError as e:
        _fail(f"Invalid JSON: {e}")


def _write_json(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        _fail(f"Failed writing output: {e}")


def cmd_ai_orphans(args):
    from ext4rescue.ai import OrphanRecord, OrphanRebuilder

    data = _load_json(args.input)
    if "orphans" not in data or not isinstance(data["orphans"], list):
        _fail("Input must contain 'orphans' array")

    if len(data["orphans"]) > AI_MAX_ITEMS:
        _fail("Too many orphan records")

    records = []
    for i, item in enumerate(data["orphans"]):
        if not isinstance(item, dict):
            _fail(f"Invalid orphan entry at index {i}")
        try:
            records.append(OrphanRecord(**item))
        except TypeError as e:
            _fail(f"Invalid orphan at index {i}: {e}")

    result = OrphanRebuilder(model=args.model).suggest_structure(records)

    if args.debug:
        print(result.raw_response, file=sys.stderr)

    _write_json(args.output, result.to_dict())


def cmd_ai_journal(args):
    from ext4rescue.ai import JournalCandidate, JournalInterpreter

    data = _load_json(args.input)
    if "items" not in data or not isinstance(data["items"], list):
        _fail("Input must contain 'items' array")

    if len(data["items"]) > AI_MAX_ITEMS:
        _fail("Too many journal items")

    items = []
    for i, item in enumerate(data["items"]):
        try:
            items.append(JournalCandidate(**item))
        except TypeError as e:
            _fail(f"Invalid journal item at {i}: {e}")

    result = JournalInterpreter(model=args.model).interpret(items)

    if args.debug:
        print(result.raw_response, file=sys.stderr)

    payload = {
        "events": [asdict(e) for e in result.events],
        "notes": result.notes,
    }
    _write_json(args.output, payload)


def cmd_ai_report(args):
    from ext4rescue.ai import ReportAI

    data = _load_json(args.input)
    result = ReportAI(model=args.model).summarize(data)

    if args.debug:
        print(result.raw_response, file=sys.stderr)

    payload = {
        "executive_summary": result.executive_summary,
        "highlights": result.highlights,
        "warnings": result.warnings,
        "next_steps": result.next_steps,
        "operator_notes": result.operator_notes,
    }
    _write_json(args.output, payload)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    p_orphans = sub.add_parser("ai-orphans")
    p_orphans.add_argument("input")
    p_orphans.add_argument("output")
    p_orphans.add_argument("--model", default="gpt-5")
    p_orphans.add_argument("--debug", action="store_true")
    p_orphans.set_defaults(func=cmd_ai_orphans)

    p_journal = sub.add_parser("ai-journal")
    p_journal.add_argument("input")
    p_journal.add_argument("output")
    p_journal.add_argument("--model", default="gpt-5")
    p_journal.add_argument("--debug", action="store_true")
    p_journal.set_defaults(func=cmd_ai_journal)

    p_report = sub.add_parser("ai-report")
    p_report.add_argument("input")
    p_report.add_argument("output")
    p_report.add_argument("--model", default="gpt-5")
    p_report.add_argument("--debug", action="store_true")
    p_report.set_defaults(func=cmd_ai_report)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
