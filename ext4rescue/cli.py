import argparse
from .scan.fs_detector import detect_filesystems
from .scan.super_hunter import hunt_superblocks


def main():
    parser = argparse.ArgumentParser(prog="ext4rescue")
    sub = parser.add_subparsers(dest="cmd")

    p_detect = sub.add_parser("detect-fs")
    p_detect.add_argument("path")

    p_hunt = sub.add_parser("hunt-super")
    p_hunt.add_argument("path")
    p_hunt.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.cmd == "detect-fs":
        res = detect_filesystems(args.path)
        for r in res:
            print(r)

    elif args.cmd == "hunt-super":
        res = hunt_superblocks(args.path)
        if args.json:
            import json
            print(json.dumps(res, indent=2))
        else:
            for r in res:
                print(r)

    else:
        parser.print_help()
