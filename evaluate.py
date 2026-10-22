"""Measure playing strength by pitting players against each other.

Every "player" here uses the same minimax search -- what differs is the
EVALUATOR it consults at the leaves:

    heuristic  minimax + your hand-written score_position()
    trained    minimax + the trained network
    untrained  minimax + an untrained network (random weights), as a floor
    random     no search at all, picks legal moves uniformly

So "trained vs heuristic" means "learned evaluator vs hand-written
evaluator, same search" -- the comparison that actually matters.

Examples:
    python evaluate.py                                    # full suite, 40 games
    python evaluate.py --p1 trained --p2 heuristic -n 3000
    python evaluate.py --p1 trained --p2 heuristic -n 500 --depth 2
    python evaluate.py -n 200                             # full suite, more games
"""

import argparse
import math
import multiprocessing
import os
import random
import time
from pathlib import Path

import torch

from connect4.game import play_game
from connect4.network import ARCHITECTURES, load_network
from connect4.players import (
    MinimaxPlayer,
    NeuralMinimaxPlayer,
    RandomOpeningPlayer,
    RandomPlayer,
)

DEFAULT_MODEL = Path("data/value_net.pt")
KINDS = ("trained", "untrained", "heuristic", "random")
GAMES_PER_WORKER = 10   # minimum work to justify spawning a process


def is_checkpoint(kind: str) -> bool:
    """Anything that isn't one of the named kinds is treated as a path to a
    saved network, so Phase B can gate models/gen3.pt against models/best.pt
    without inventing a name for every generation."""
    return kind not in KINDS


def build_player(kind, depth, random_moves, rng, nets):
    """nets maps a kind (or checkpoint path) to an already-loaded network, so
    the expensive load happens once per worker rather than once per game."""
    if kind == "random":
        return RandomPlayer(rng)
    if kind == "heuristic":
        inner = MinimaxPlayer(depth=depth)
    elif kind in nets:
        # Evaluation must be deterministic: no epsilon here, ever. Exploration
        # belongs in data generation, not in measurement.
        inner = NeuralMinimaxPlayer(nets[kind], depth=depth)
    else:
        raise ValueError(f"unknown player kind: {kind}")
    # Random openings: minimax and the network are both deterministic, so
    # without them every game between the same two players is identical.
    return RandomOpeningPlayer(inner, random_moves=random_moves, rng=rng)


def _play_chunk(task):
    """Play a range of games in a worker process. Returns (wins, draws,
    losses) from p1_kind's perspective."""
    (start, count, p1_kind, p2_kind, depth, random_moves,
     model_path, base_seed, arch_name) = task

    # One torch thread per worker -- see the note in main.py._play_chunk.
    torch.set_num_threads(1)

    nets = {}
    for kind in (p1_kind, p2_kind):
        if kind in nets:
            continue
        if kind == "trained":
            nets[kind] = load_network(model_path)
        elif kind == "untrained":
            torch.manual_seed(0)  # fixed, so this floor doesn't move between runs
            nets[kind] = ARCHITECTURES[arch_name]()
            nets[kind].eval()
        elif is_checkpoint(kind):
            nets[kind] = load_network(kind)

    wins = draws = losses = 0
    for i in range(start, start + count):
        rng = random.Random(base_seed + i)
        a = build_player(p1_kind, depth, random_moves, rng, nets)
        b = build_player(p2_kind, depth, random_moves, rng, nets)

        # Alternate who moves first so first-player advantage cancels out.
        a_first = i % 2 == 0
        first, second = (a, b) if a_first else (b, a)
        board, _ = play_game(first, second, verbose=False)

        if board.winner is None:
            draws += 1
        elif (board.winner == 1) == a_first:
            wins += 1
        else:
            losses += 1

    return wins, draws, losses


def duel(label, p1_kind, p2_kind, args, arch_name):
    """Run one matchup in parallel and print the result with error bars."""
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    # Each worker is a fresh process that has to import torch (several
    # seconds on Windows, which spawns rather than forks). Only start one
    # per ~25 games, or startup costs more than the work it saves.
    workers = max(1, min(workers, args.games // GAMES_PER_WORKER))

    chunk = max(1, min(20, args.games // (workers * 4) or 1))
    tasks = []
    played = 0
    while played < args.games:
        n = min(chunk, args.games - played)
        tasks.append((played, n, p1_kind, p2_kind, args.depth, args.random_moves,
                      args.model, args.seed, arch_name))
        played += n

    wins = draws = losses = 0
    pool = None
    if workers == 1:
        results = map(_play_chunk, tasks)
    else:
        pool = multiprocessing.Pool(processes=workers)
        results = pool.imap_unordered(_play_chunk, tasks)

    try:
        for w, d, l in results:
            wins += w
            draws += d
            losses += l
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    n = wins + draws + losses
    score = (wins + 0.5 * draws) / n
    # Standard error of the score, so a real difference is distinguishable
    # from sampling noise. 40 games is roughly +/- 15 points; 1000 is +/- 3.
    se = math.sqrt(max(score * (1 - score), 1e-9) / n)

    print(f"{label:<36} {wins:>5}W {draws:>4}D {losses:>5}L   "
          f"score {100 * score:5.1f}% +/- {100 * 1.96 * se:.1f}")
    return score


def main():
    parser = argparse.ArgumentParser(
        description="Measure playing strength",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="player kinds: " + ", ".join(KINDS))
    parser.add_argument("--p1", default=None,
                        help="First player: one of " + ", ".join(KINDS) +
                             ", or a path to a saved checkpoint")
    parser.add_argument("--p2", default=None,
                        help="Second player: a kind or a checkpoint path")
    parser.add_argument("-n", "--games", type=int, default=40)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--random-moves", type=int, default=6, dest="random_moves")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; the same value replays the same set of games")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel processes (0 = all cores but one)")
    args = parser.parse_args()

    sides = [k for k in (args.p1, args.p2) if k]
    for k in sides:
        if is_checkpoint(k) and not Path(k).exists():
            raise SystemExit(f"No such checkpoint: {k}")

    # --model is only needed when a side actually says "trained" (or when the
    # full suite runs, which always does).
    needs_model = not sides or "trained" in sides or "untrained" in sides
    if needs_model and not args.model.exists():
        raise SystemExit(f"No trained model at {args.model}. Run `python train.py` first.")

    reference = args.model if needs_model else Path(next(k for k in sides if is_checkpoint(k)))
    arch_name = ("conv" if type(load_network(reference)).__name__ == "ConvValueNetwork"
                 else "dense")

    print(f"{args.games} games per matchup, search depth {args.depth}")
    print("(a draw counts as half a win; +/- is a 95% confidence interval)\n")

    start = time.time()
    if args.p1 or args.p2:
        p1 = args.p1 or "trained"
        p2 = args.p2 or "heuristic"
        duel(f"{p1} vs {p2}", p1, p2, args, arch_name)
    else:
        duel("trained    vs random", "trained", "random", args, arch_name)
        duel("untrained  vs random", "untrained", "random", args, arch_name)
        duel("heuristic  vs random", "heuristic", "random", args, arch_name)
        print()
        duel("trained    vs heuristic   <- the test", "trained", "heuristic", args, arch_name)
        duel("trained    vs untrained", "trained", "untrained", args, arch_name)

    print(f"\n{time.time() - start:.0f}s")


if __name__ == "__main__":
    main()