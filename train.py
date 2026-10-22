"""Train the ValueNetwork on recorded self-play games.

--model-out is required: training overwrites it, and an accidental default
would silently destroy whatever network happens to be at that path.

Examples:
    python train.py --model-out models/gen1.pt
    python train.py --model-out models/gen1.pt --data data/minimax_games.jsonl data/gen1.jsonl
    python train.py --model-out models/gen1.pt --epochs 15 --lr 0.0005
    python train.py --model-out models/gen1.pt --cpu   # skip DirectML
"""

import argparse
import hashlib
import math
from pathlib import Path

import torch
import torch.nn as nn

from connect4.network import (ARCHITECTURES, boards_to_tensor, load_network,
                              save_network)
from connect4.recorder import load_games

DEFAULT_DATA = [Path("data/minimax_games.jsonl")]
CACHE_DIR = Path("data/.tensor_cache")
KEEP_CACHES = 2   # stale caches are a few hundred MB each
CONVERT_CHUNK = 250_000   # positions converted per vectorised call

# How much total training to do, measured in "example presentations"
# (positions seen, counting repeats across epochs). This is what 30 epochs
# over ~130k positions amounted to. Deliberately NOT measured in weight
# updates, which would change meaning whenever batch size changes.
TARGET_EXAMPLES = 4_000_000
MAX_AUTO_EPOCHS = 60     # guard: tiny datasets would otherwise ask for thousands


def expand_paths(paths):
    """Turn what the user typed into a concrete list of .jsonl files.

    Accepts individual files, directories (every .jsonl inside), and glob
    patterns like data/selfplay_gen*.jsonl -- PowerShell doesn't expand
    globs for us the way a Unix shell would, so we do it here."""
    resolved = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.glob("*.jsonl"))
            if not found:
                raise SystemExit(f"No .jsonl files in directory: {p}")
            resolved.extend(found)
        elif any(ch in str(raw) for ch in "*?["):
            found = sorted(Path().glob(str(raw)))
            if not found:
                raise SystemExit(f"Pattern matched no files: {raw}")
            resolved.extend(found)
        elif p.exists():
            resolved.append(p)
        else:
            raise SystemExit(f"No such data file: {p}")

    # Drop duplicates while preserving order (e.g. a file named twice).
    seen = set()
    unique = []
    for p in resolved:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def cache_key(files) -> str:
    """Fingerprint the exact data being loaded. Any change to which files are
    used, or to their size or modification time, produces a different key --
    so appending new games automatically invalidates the old cache."""
    # "v2" is a cache format marker -- bumping it invalidates older caches
    # that predate the game-id field.
    parts = ["v3"]
    for p in files:
        st = p.stat()
        parts.append(f"{p.resolve()}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def build_dataset(files):
    """Convert recorded games into tensors. This is the slow part -- a Python
    loop over every cell of every position -- which is why we cache it.

    Also returns a game id per position, so the train/validation split can be
    made by GAME. Splitting by position would scatter positions from the same
    game across both sets; since they all share one outcome label, the network
    could score well on validation by memorizing game outcomes rather than
    learning anything that generalizes."""
    chunks = []                 # converted uint8 tensors, concatenated at the end
    grids, players = [], []     # positions awaiting conversion
    targets = []
    groups = []
    remaining = []
    game_id = 0
    n_positions = 0

    def flush():
        """Convert everything buffered so far in one vectorised call. Done in
        batches rather than all at once so peak memory stays bounded -- the
        full dataset as raw Python lists would be many GB."""
        if grids:
            chunks.append(boards_to_tensor(grids, players))
            grids.clear()
            players.clear()

    for path in files:
        before = n_positions
        for game in load_games(path):
            n_moves = len(game["moves"])
            for i, move in enumerate(game["moves"]):
                grids.append(move["board_before"])
                players.append(move["player"])
                targets.append(move["value_target"])
                groups.append(game_id)
                remaining.append(n_moves - i)   # plies until the game ends
                n_positions += 1
            game_id += 1
            if len(grids) >= CONVERT_CHUNK:
                flush()
        print(f"  {path}: {n_positions - before} positions")

    flush()
    x = torch.cat(chunks).float()
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    g = torch.tensor(groups, dtype=torch.long)
    r = torch.tensor(remaining, dtype=torch.long)
    return x, y, g, r


def load_dataset(paths, use_cache: bool = True):
    """Turn recorded games from one or more JSONL files into
    (inputs, targets, game_ids, moves_remaining), reusing a cached conversion when the
    underlying files haven't changed."""
    files = expand_paths(paths)

    if not use_cache:
        return build_dataset(files)

    cache_file = CACHE_DIR / f"{cache_key(files)}.pt"
    if cache_file.exists():
        blob = torch.load(cache_file, map_location="cpu", weights_only=True)
        # Boards are stored as uint8 (every value is 0 or 1) to keep the cache
        # ~4x smaller on disk; the network wants float32.
        print(f"  loaded cached tensors ({cache_file.name})")
        return blob["x"].float(), blob["y"], blob["g"], blob["r"]

    x, y, g, r = build_dataset(files)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x.to(torch.uint8), "y": y, "g": g, "r": r}, cache_file)
    print(f"  cached for next time ({cache_file.name})")
    prune_cache(keep=KEEP_CACHES)
    return x, y, g, r


def prune_cache(keep: int = 2) -> None:
    """Delete stale cache files. Every time games are added the key changes and
    a new cache is written, orphaning the old one -- at a few hundred MB each
    these pile up fast. Keeps the most recently used ones in case you switch
    back and forth between datasets."""
    caches = sorted(CACHE_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    freed = 0
    for old in caches[keep:]:
        freed += old.stat().st_size
        old.unlink()
    if freed:
        print(f"  pruned {len(caches) - keep} stale cache file(s), freed {freed / 1e6:.0f} MB")


def evaluate_loss(net, x, y, loss_fn, device, batch_size):
    """Average loss over a dataset, without training on it."""
    net.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            bx = x[start:start + batch_size].to(device)
            by = y[start:start + batch_size].to(device)
            total += loss_fn(net(bx), by).item()
            batches += 1
    net.train()
    return total / max(1, batches)


def train(args):
    print("Loading data...")
    x, y, groups, remaining = load_dataset(args.data, use_cache=not args.no_cache)
    print(f"{len(x)} positions from {groups.max().item() + 1} games, "
          f"input shape {tuple(x.shape)}")

    # Discount distant outcomes. A position 30 moves from the end says much
    # less about who deserved to win than one 2 moves from the end, but the
    # raw label claims +1/-1 with equal confidence either way. Scaling by
    # discount^(moves remaining) pulls far-from-the-end labels toward 0
    # (i.e. "uncertain"), which is the honest claim.
    if args.discount < 1.0:
        y = y * (args.discount ** remaining.unsqueeze(1).float())
        print(f"discount {args.discount}: labels scaled by "
              f"{args.discount}^(moves remaining)")

    if args.cpu:
        device = torch.device("cpu")
    else:
        import torch_directml
        device = torch_directml.device()
    print(f"training on: {device}")

    # Split by GAME, not by position. Positions from one game share an
    # outcome label, so a position-level split would let the network score
    # well on validation by memorizing game outcomes instead of generalizing.
    if args.val_frac > 0:
        game_ids = groups.unique()
        shuffled = game_ids[torch.randperm(len(game_ids))]
        n_val = max(1, int(len(shuffled) * args.val_frac))
        val_games = shuffled[:n_val]

        is_val = torch.isin(groups, val_games)
        x_train, y_train = x[~is_val], y[~is_val]
        x_val, y_val = x[is_val], y[is_val]
        print(f"split: {len(x_train):,} train / {len(x_val):,} validation "
              f"positions ({n_val} of {len(game_ids)} games held out)")
    else:
        x_train, y_train = x, y
        x_val = y_val = None
        print("no validation split -- keeping the final epoch's weights")

    # 'auto': do a fixed amount of total training as the dataset grows, so
    # wall-clock stays flat instead of climbing. Measured in examples seen,
    # which (unlike a target update count) doesn't shift when batch size does.
    batches_per_epoch = max(1, len(x_train) // args.batch_size)
    if args.epochs == "auto":
        args.epochs = min(MAX_AUTO_EPOCHS, max(1, round(TARGET_EXAMPLES / len(x_train))))
        print(f"auto epochs: {args.epochs} "
              f"(~{args.epochs * len(x_train):,} examples seen, "
              f"{args.epochs * batches_per_epoch:,} updates)")
    else:
        args.epochs = int(args.epochs)

    if args.init_from:
        # Warm start. Phase B refines an already-good network rather than
        # rediscovering the game each generation, so pair this with a lower
        # --lr: the default rate would knock it off what it already knows.
        net = load_network(args.init_from).to(device)
        net.train()
        loaded = next(name for name, cls in ARCHITECTURES.items() if isinstance(net, cls))
        if loaded != args.arch:
            print(f"note: --arch {args.arch} ignored, {args.init_from} is {loaded}")
        args.arch = loaded
        print(f"warm start from {args.init_from}")
    else:
        net = ARCHITECTURES[args.arch]().to(device)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"architecture: {args.arch} ({n_params:,} weights)")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    best_val = math.inf

    for epoch in range(args.epochs):
        # Shuffle each epoch so the network doesn't pick up on the order
        # examples happen to arrive in.
        perm = torch.randperm(len(x_train))
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        total_loss = 0.0
        batches = 0

        for start in range(0, len(x_train), args.batch_size):
            batch_x = x_shuffled[start:start + args.batch_size].to(device)
            batch_y = y_shuffled[start:start + args.batch_size].to(device)

            optimizer.zero_grad()
            predictions = net(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        train_loss = total_loss / batches

        if x_val is None:
            print(f"epoch {epoch + 1}/{args.epochs}  train {train_loss:.4f}")
            save_network(net, args.model_out)
            continue

        val_loss = evaluate_loss(net, x_val, y_val, loss_fn, device, args.batch_size)

        # Only keep a checkpoint that actually improved on held-out data.
        # Once validation stops improving while training loss keeps falling,
        # the extra epochs are memorizing rather than learning.
        if val_loss < best_val:
            best_val = val_loss
            save_network(net, args.model_out)
            marker = "  <- saved"
        else:
            marker = ""

        print(f"epoch {epoch + 1}/{args.epochs}  "
              f"train {train_loss:.4f}  val {val_loss:.4f}{marker}")

    if x_val is not None:
        print(f"Done. Best validation loss {best_val:.4f}, weights at {args.model_out}")
    else:
        print(f"Done. Weights saved to {args.model_out}")


def main():
    parser = argparse.ArgumentParser(description="Train the Connect 4 value network")
    parser.add_argument("--data", type=Path, nargs="+", default=DEFAULT_DATA,
                        help="One or more JSONL game files to train on")
    parser.add_argument("--model-out", type=Path, default=None, dest="model_out",
                        help="Where to save the trained weights (REQUIRED -- this path "
                             "gets overwritten, so it must be a deliberate choice)")
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="conv",
                        help="Network architecture (conv is much better at board patterns)")
    parser.add_argument("--epochs", default="auto",
                        help="Number of passes over the data, or 'auto' to size it so total "
                             "training stays constant as the dataset grows")
    parser.add_argument("--batch-size", type=int, default=512, dest="batch_size")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate (scaled up to suit the larger default batch)")
    parser.add_argument("--val-frac", type=float, default=0.1, dest="val_frac",
                        help="Fraction of GAMES held out to pick the best checkpoint "
                             "(0 disables, keeping the last epoch)")
    parser.add_argument("--discount", type=float, default=1.0,
                        help="Scale labels by discount^(moves remaining). 1.0 keeps raw "
                             "game outcomes; ~0.95-0.98 makes distant positions claim less")
    parser.add_argument("--init-from", type=Path, default=None, dest="init_from",
                        help="Start from these weights instead of random ones "
                             "(Phase B: models/best.pt). Use a lower --lr with it.")
    parser.add_argument("--no-cache", action="store_true", dest="no_cache",
                        help="Rebuild tensors from JSON instead of using the cache")
    parser.add_argument("--cpu", action="store_true",
                        help="Train on CPU instead of the DirectML GPU")
    args = parser.parse_args()

    # Deliberately not an argparse default. Training overwrites this path, and
    # the old default (data/value_net.pt) held the only copy of a network that
    # took hours to produce -- one absent-minded `python train.py` would have
    # destroyed it. Checked here rather than in train() so it fails instantly
    # instead of after the data load.
    if args.model_out is None:
        raise SystemExit(
            """error: --model-out is required.
  Training overwrites this file, so it has to be a deliberate choice.
  Phase B:  --model-out models/gen<N>.pt
  Never:    --model-out models/best.pt   (promote with `copy` after gating)
  Never:    --model-out models/gen0.pt   (frozen baseline)"""
        )

    train(args)


if __name__ == "__main__":
    main()