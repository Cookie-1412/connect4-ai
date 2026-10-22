"""Remove duplicate games from recorded game files.

Two games count as duplicates if they have the same move sequence. This
happens because minimax and the network are both deterministic: once two
games share an opening, everything after it is identical. Duplicates aren't
corrupt data, but they silently over-weight those positions during training.

Files are processed as one stream, so duplicates ACROSS files are removed
too (useful when a later self-play generation replays games from an
earlier one). The first occurrence wins.

Examples:
    python dedupe.py data/minimax_games.jsonl
    python dedupe.py "data/selfplay_gen*.jsonl"
    python dedupe.py data/ --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path


def expand_paths(paths):
    """Accept files, directories, or glob patterns (PowerShell won't expand
    globs for us, so we do it here)."""
    resolved = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            resolved.extend(sorted(p.glob("*.jsonl")))
        elif any(ch in str(raw) for ch in "*?["):
            resolved.extend(sorted(Path().glob(str(raw))))
        elif p.exists():
            resolved.append(p)
        else:
            raise SystemExit(f"No such file: {p}")

    seen = set()
    unique = []
    for p in resolved:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if not unique:
        raise SystemExit("No .jsonl files matched.")
    return unique


def signature(game: dict) -> tuple:
    """What makes two games 'the same': their move sequence."""
    return tuple(move["action"] for move in game["moves"])


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate games from JSONL files")
    parser.add_argument("paths", nargs="+", help="Files, directories, or glob patterns")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be removed without changing anything")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip writing a .bak copy before rewriting")
    args = parser.parse_args()

    files = expand_paths(args.paths)
    seen = set()
    total_before = total_after = 0

    for path in files:
        kept_lines = []
        n_before = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_before += 1
                sig = signature(json.loads(line))
                if sig in seen:
                    continue
                seen.add(sig)
                kept_lines.append(line)

        n_after = len(kept_lines)
        removed = n_before - n_after
        total_before += n_before
        total_after += n_after

        pct = (100 * removed / n_before) if n_before else 0
        print(f"{path}: {n_before} -> {n_after} games ({removed} duplicates, {pct:.0f}%)")

        if args.dry_run or removed == 0:
            continue

        if not args.no_backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

        with path.open("w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")

    print()
    if args.dry_run:
        print(f"DRY RUN -- nothing changed. Would keep {total_after} of {total_before} games.")
    else:
        print(f"Kept {total_after} of {total_before} games.")
        if not args.no_backup and total_after != total_before:
            print("Originals backed up alongside each file with a .bak extension.")


if __name__ == "__main__":
    main()
