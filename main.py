"""Connect 4 CLI: play a game, or generate self-play training data.

Examples:
    python main.py play                     # human vs human
    python main.py play --vs random          # human vs random AI
    python main.py selfplay -n 500           # 500 random-vs-random games -> data/games.jsonl
    python main.py export-npz                # flatten data/games.jsonl -> data/games.npz
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import time
from pathlib import Path

import torch

from connect4.game import play_game
from connect4.network import load_network
from connect4.players import (
    HumanPlayer,
    MinimaxPlayer,
    NeuralMinimaxPlayer,
    RandomOpeningPlayer,
    RandomPlayer,
)
from connect4.recorder import export_to_numpy

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_JSONL = DATA_DIR / "games.jsonl"
MINIMAX_JSONL = DATA_DIR / "minimax_games.jsonl"
MODEL_FILE = DATA_DIR / "value_net.pt"


def cmd_play(args: argparse.Namespace) -> None:
    p1 = HumanPlayer()
    p2 = RandomPlayer() if args.vs == "random" else HumanPlayer()
    _, recorder = play_game(p1, p2, verbose=True, record=args.record)
    if recorder:
        recorder.append_to_jsonl(DEFAULT_JSONL)
        print(f"\nGame recorded to {DEFAULT_JSONL}")


def make_player(kind: str, depth: int, random_moves: int, rng: random.Random,
                network=None, epsilon: float = 0.0, sample_plies: int = 0):
    """Build one player from a CLI name. Non-random players get random
    openings so repeated games aren't all identical (minimax and the
    network are both deterministic on their own).

    epsilon/sample_plies add exploration *during* the game for the neural
    player. That is a better source of variety than a long random opening,
    which starts games from positions strong play would never reach."""
    if kind == "random":
        return RandomPlayer(rng)
    if kind == "minimax":
        inner = MinimaxPlayer(depth=depth)
    elif kind == "neural":
        inner = NeuralMinimaxPlayer(network, depth=depth, epsilon=epsilon,
                                    sample_plies=sample_plies, rng=rng)
    else:
        raise ValueError(f"unknown player kind: {kind}")
    return RandomOpeningPlayer(inner, random_moves=random_moves, rng=rng)

def count_existing_games(path: Path) -> int:
    """How many games are already in this file. Counts lines rather than
    parsing JSON: each game is exactly one line, and parsing a 90k-game file
    would take ~10s for a number we only need approximately."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def pick_random_moves(existing: int, threshold: float = 0.15, cap: int = 10) -> int:
    """Smallest opening length whose pool is large enough that new games rarely
    collide with existing ones. After the random opening both players are
    deterministic, so a repeated opening yields a byte-identical game."""
    for candidate in range(6, cap + 1):
        if existing / (7 ** candidate) < threshold:
            return candidate
    return cap

def _play_chunk(task):
    """Play a batch of games. Runs inside a worker process, so everything it
    needs arrives through `task` (Windows spawns fresh processes rather than
    forking, so nothing is inherited from the parent).

    Returns finished games as JSON strings -- the parent does all file
    writing, which avoids several processes appending to one file at once."""
    (n_games, p1_kind, p2_kind, depth, random_moves, model_path, seed,
     epsilon, sample_plies) = task

    # One torch thread per worker. Torch otherwise sizes its pool to the whole
    # machine, so N workers each spawn N threads and they spend their time
    # fighting rather than working -- measured 4.3x slower at 24 workers.
    torch.set_num_threads(1)

    # Each worker MUST have its own seed. Sharing one would make every
    # worker replay the identical set of games.
    rng = random.Random(seed)

    network = None
    if "neural" in (p1_kind, p2_kind):
        network = load_network(model_path)

    lines = []
    for _ in range(n_games):
        p1 = make_player(p1_kind, depth, random_moves, rng, network, epsilon, sample_plies)
        p2 = make_player(p2_kind, depth, random_moves, rng, network, epsilon, sample_plies)
        _, recorder = play_game(p1, p2, verbose=False, record=True)
        lines.append(json.dumps(recorder.record.to_dict()))
    return lines


def cmd_selfplay(args: argparse.Namespace) -> None:
    # --p1/--p2 override --player; --player sets both sides when they're absent.
    p1_kind = args.p1 or args.player
    p2_kind = args.p2 or args.player

    out_path = args.out
    if out_path is None:
        both_random = p1_kind == "random" and p2_kind == "random"
        out_path = DEFAULT_JSONL if both_random else MINIMAX_JSONL

    if args.random_moves is None:
        existing = count_existing_games(out_path)
        args.random_moves = pick_random_moves(existing)
        opening_pool = 7 ** args.random_moves
        print(f"auto random-moves {args.random_moves} "
              f"({existing:,} existing games, pool {opening_pool:,}, "
              f"{100 * existing / opening_pool:.0f}% saturated)")

    # Workers each load the model themselves; just check it exists up front
    # so we fail immediately rather than inside every worker.
    if "neural" in (p1_kind, p2_kind) and not args.model.exists():
        raise SystemExit(f"No trained model at {args.model}. Run `python train.py` first.")

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, args.num_games)

    print(f"{p1_kind} (P1) vs {p2_kind} (P2) -> {out_path}")
    print(f"{args.num_games} games across {workers} worker{'s' if workers > 1 else ''}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Split into many small chunks rather than one per worker: games vary in
    # length, so small chunks keep every core busy instead of leaving some
    # idle waiting on a straggler.
    chunk_size = max(1, min(25, args.num_games // (workers * 4) or 1))
    chunks = []
    remaining = args.num_games
    while remaining > 0:
        chunks.append(min(chunk_size, remaining))
        remaining -= chunk_size

    # Give every chunk its own seed. Without this, workers would each replay
    # the same games. A fixed --seed keeps the whole run reproducible.
    base = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    tasks = [
        (n, p1_kind, p2_kind, args.depth, args.random_moves, args.model, base + i,
         args.epsilon, args.sample_plies)
        for i, n in enumerate(chunks)
    ]

    start = time.time()
    done = 0
    report_every = max(1, args.num_games // 20)
    next_report = report_every

    with open(out_path, "a", encoding="utf-8") as f:
        if workers == 1:
            results = map(_play_chunk, tasks)
        else:
            pool = multiprocessing.Pool(processes=workers)
            results = pool.imap_unordered(_play_chunk, tasks)

        try:
            for lines in results:
                for line in lines:
                    f.write(line + "\n")
                done += len(lines)
                if done >= next_report:
                    rate = done / max(1e-9, time.time() - start)
                    eta = (args.num_games - done) / rate
                    hours, rem = divmod(int(eta), 3600)
                    mins = rem // 60
                    left = f"{hours}h {mins}m" if hours else f"{mins} min"
                    print(f"{done}/{args.num_games} games  ({rate:.1f}/s, ~{left} left)")
                    next_report += report_every
        except KeyboardInterrupt:
            print("\ninterrupted -- games completed so far are saved")
            if workers > 1:
                pool.terminate()
            raise
        finally:
            if workers > 1:
                pool.close()
                pool.join()

    elapsed = time.time() - start
    print(f"Done. {done} games in {elapsed/60:.1f} min "
          f"({done/max(1e-9, elapsed):.1f}/s) -> {out_path}")


def cmd_export_npz(args: argparse.Namespace) -> None:
    export_to_numpy(args.jsonl, args.out)
    print(f"Exported {args.jsonl} -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect 4")
    sub = parser.add_subparsers(dest="command", required=True)

    p_play = sub.add_parser("play", help="Play an interactive game")
    p_play.add_argument("--vs", choices=["human", "random"], default="human")
    p_play.add_argument("--record", action="store_true", help="Save this game to data/games.jsonl")
    p_play.set_defaults(func=cmd_play)

    p_self = sub.add_parser("selfplay", help="Generate training data via self-play")
    p_self.add_argument("-n", "--num-games", type=int, default=100, dest="num_games")
    p_self.add_argument("--seed", type=int, default=None)
    p_self.add_argument("--player", choices=["random", "minimax", "neural"], default="random",
                        help="Player type for BOTH sides (overridden by --p1/--p2)")
    p_self.add_argument("--p1", choices=["random", "minimax", "neural"], default=None,
                        help="Player type for player 1 only")
    p_self.add_argument("--p2", choices=["random", "minimax", "neural"], default=None,
                        help="Player type for player 2 only")
    p_self.add_argument("--depth", type=int, default=3)
    p_self.add_argument("--random-moves", type=int, default=None, dest="random_moves",
                        help="Random opening moves before real play starts. More = more "
                             "diverse games but noisier labels (7^N possible openings).")
    p_self.add_argument("--workers", type=int, default=0,
                        help="Parallel processes (0 = all cores but one, 1 = no multiprocessing)")
    p_self.add_argument("--model", type=Path, default=MODEL_FILE,
                        help="Checkpoint the 'neural' player uses (Phase B: models/best.pt)")
    p_self.add_argument("--epsilon", type=float, default=0.0,
                        help="Explore among moves within this much of the best score. "
                             "0 = always play the strongest move.")
    p_self.add_argument("--sample-plies", type=int, default=0, dest="sample_plies",
                        help="Explore only for this many opening plies, then play strongest. "
                             "Sampling late corrupts the outcome label.")
    p_self.add_argument("--out", type=Path, default=None)
    p_self.set_defaults(func=cmd_selfplay)

    p_export = sub.add_parser("export-npz", help="Flatten a JSONL game log into a .npz tensor file")
    p_export.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    p_export.add_argument("--out", type=Path, default=DATA_DIR / "games.npz")
    p_export.set_defaults(func=cmd_export_npz)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()