#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from ext4rescue.scan.fs_detector import detect_filesystem
from ext4rescue.scan.super_hunter import hunt_superblocks
from ext4rescue.ext4.recover import run_recovery
from ext4rescue.carve.engine import run_carving
from ext4rescue.report.html import generate_report

AI_MAX_ITEMS = 300


def _fail(msg: str):
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(p, d):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def cmd_detect_fs(a):
    with open(a.input, "rb") as f:
        data = f.read(8 * 1024 * 1024)
    res = detect_filesystem(data, data[-1024:])
    print(json.dumps(res, indent=2))


def cmd_hunt_super(a):
    res = hunt_superblocks(a.input)
    print(json.dumps([r.to_dict() for r in res], indent=2))


def cmd_recover(a):
    r = run_recovery(a.input, a.output, sb_offset=a.sb_offset)
    print(json.dumps(r.to_dict(), indent=2))


def cmd_carve(a):
    r = run_carving(a.input, a.output)
    print(json.dumps(r.to_dict(), indent=2))


def cmd_report(a):
    data = _load_json(a.input)

    class _Obj:
        def __init__(self, d):
            self._d = d

        def to_dict(self):
            return self._d

    html = generate_report(data.get("disk_path", ""), _Obj(data), None)
    with open(a.output, "w", encoding="utf-8") as f:
        f.write(html)


def cmd_ai_orphans(a):
    from ext4rescue.ai import OrphanRecord, OrphanRebuilder

    d = _load_json(a.input)
    items = [OrphanRecord(**x) for x in d["orphans"]]
    r = OrphanRebuilder(model=a.model).suggest_structure(items)
    if a.debug:
        print(r.raw_response, file=sys.stderr)
    _write_json(a.output, r.to_dict())


def cmd_ai_journal(a):
    from ext4rescue.ai import JournalCandidate, JournalInterpreter

    d = _load_json(a.input)
    items = [JournalCandidate(**x) for x in d["items"]]
    r = JournalInterpreter(model=a.model).interpret(items)
    if a.debug:
        print(r.raw_response, file=sys.stderr)
    _write_json(a.output, {"events": [asdict(e) for e in r.events], "notes": r.notes})


def cmd_ai_report(a):
    from ext4rescue.ai import ReportAI

    d = _load_json(a.input)
    r = ReportAI(model=a.model).summarize(d)
    if a.debug:
        print(r.raw_response, file=sys.stderr)
    _write_json(a.output, r.to_dict())


def main():
    p = argparse.ArgumentParser(prog="ext4rescue")
    sub = p.add_subparsers(dest="cmd")

    for name, fn in [
        ("detect-fs", cmd_detect_fs),
        ("hunt-super", cmd_hunt_super),
        ("recover-named", cmd_recover),
        ("carve", cmd_carve),
        ("report", cmd_report),
    ]:
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("output", nargs="?")
        if name == "recover-named":
            sp.add_argument("--sb-offset", type=int)
        sp.set_defaults(func=fn)

    for name, fn in [
        ("ai-orphans", cmd_ai_orphans),
        ("ai-journal", cmd_ai_journal),
        ("ai-report", cmd_ai_report),
    ]:
        sp = sub.add_parser(name)
        sp.add_argument("input")
        sp.add_argument("output")
        sp.add_argument("--model", default="gpt-5")
        sp.add_argument("--debug", action="store_true")
        sp.set_defaults(func=fn)

    a = p.parse_args()
    if not hasattr(a, "func"):
        p.print_help()
        return
    a.func(a)


if __name__ == "__main__":
    main()
