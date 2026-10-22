"""Report what's in your recorded game files, without training on them.

Examples:
    python stats.py                                  # all .jsonl in data/
    python stats.py data/minimax_games.jsonl
    python stats.py "data/selfplay_gen*.jsonl"
"""

import argparse
from collections import Counter
from pathlib import Path

from connect4.recorder import load_games
# Imported rather than copied: these used to be duplicated here and drifted out
# of sync with train.py, so the epoch advice described a run you'd never do.
from train import MAX_AUTO_EPOCHS, TARGET_EXAMPLES

DEFAULT_BATCH_SIZE = 512   # train.py's --batch-size default
DEFAULT_VAL_FRAC = 0.1     # train.py's --val-frac default; those games aren't trained on


def expand_paths(paths):
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

    seen, unique = set(), []
    for p in resolved:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    if not unique:
        raise SystemExit("No .jsonl files matched.")
    return unique


def main():
    parser = argparse.ArgumentParser(description="Summarize recorded game data")
    parser.add_argument("paths", nargs="*", default=["data"],
                        help="Files, directories, or glob patterns (default: data/)")
    args = parser.parse_args()

    total_games = total_positions = 0
    all_signatures = set()
    all_outcomes = Counter()

    print(f"{'file':<38} {'games':>7} {'unique':>7} {'positions':>10}")
    print("-" * 66)

    for path in expand_paths(args.paths):
        games = list(load_games(path))
        sigs = {tuple(m["action"] for m in g["moves"]) for g in games}
        positions = sum(len(g["moves"]) for g in games)

        total_games += len(games)
        total_positions += positions
        all_signatures |= sigs
        all_outcomes.update(g["outcome"] for g in games)

        print(f"{str(path):<38} {len(games):>7} {len(sigs):>7} {positions:>10,}")

    print("-" * 66)
    print(f"{'TOTAL':<38} {total_games:>7} {len(all_signatures):>7} {total_positions:>10,}")
    print()

    if total_games:
        dupes = total_games - len(all_signatures)
        print(f"duplicates: {dupes} ({100 * dupes / total_games:.0f}%)"
              + ("  -> worth running dedupe.py" if dupes > total_games * 0.05 else "  -> fine"))
        print(f"outcomes:   {dict(all_outcomes)}")

    if total_positions:
        # Mirror train.py's --epochs auto exactly: it sizes epochs so that the
        # TOTAL number of examples seen stays constant as the dataset grows,
        # and it sees only the training split, not the held-out games.
        train_positions = max(1, round(total_positions * (1 - DEFAULT_VAL_FRAC)))
        suggested = min(MAX_AUTO_EPOCHS, max(1, round(TARGET_EXAMPLES / train_positions)))
        batches_per_epoch = max(1, train_positions // DEFAULT_BATCH_SIZE)
        print()
        print(f"{train_positions:,} training positions after a "
              f"{100 * DEFAULT_VAL_FRAC:.0f}% validation split")
        print(f"{batches_per_epoch:,} batches per epoch at batch size {DEFAULT_BATCH_SIZE}")
        print(f"--epochs auto will pick {suggested}   "
              f"(~{suggested * train_positions:,} examples seen, "
              f"{suggested * batches_per_epoch:,} weight updates)")
        if suggested == MAX_AUTO_EPOCHS:
            print(f"  (capped at MAX_AUTO_EPOCHS={MAX_AUTO_EPOCHS})")


if __name__ == "__main__":
    main()