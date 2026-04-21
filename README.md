# Recovery_HDD

`Recovery_HDD` is a Linux-first Python CLI for recovering data from an ext4 disk that was later overwritten by ZFS labels and/or a second quick format.

The project prioritizes filesystem-aware recovery before carving:

1. detect layered filesystem signatures
2. hunt ext4 backup superblocks
3. score and validate candidates
4. recover files with original names and paths where possible
5. carve remaining files as a fallback

## Current status

Sprint 1 provides the foundation:

- modern Python packaging
- CLI entry points
- safe low-level I/O helpers built around `os.pread`
- raw filesystem signature detection for ext4, NTFS, FAT32, and ZFS traces
- ext4 superblock parsing and scoring
- candidate backup superblock generation
- a reproducible disk-lab script for testing
- first unit tests for superblock candidate logic

## Planned commands

```bash
python -m ext4rescue detect-fs disk.img
python -m ext4rescue hunt-super disk.img --json
python -m ext4rescue recover-named disk.img out/
python -m ext4rescue carve disk.img out/
```

## Design principles

- read-only by default
- evidence-based scoring, not magic guesses
- sequential I/O for spinning disks
- modular architecture so ext4 parsing, carving, reporting, and AI-assisted organization stay separate
- JSON-friendly outputs so later phases can reuse previous scan results

## Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

python -m ext4rescue --help
python -m ext4rescue detect-fs /path/to/disk.img
python -m ext4rescue hunt-super /path/to/disk.img --json
```

## Safety notes

- Work on a disk image whenever possible.
- Prefer `ddrescue` over direct work on the original drive.
- Open block devices read-only.
- Do not run repair tools that modify metadata on the original evidence.

## Repository layout

```text
ext4rescue/
  scan/      layered signature detection and candidate generation
  ext4/      ext4 structures and recovery logic
  carve/     fallback file carving
  report/    HTML reports and statistics
  ai/        optional AI-assisted post-processing
```

## License

Initial development version. Add a project license before public release.
