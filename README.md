# Connect 4 AI

A Connect 4 engine that learns to evaluate positions from self-play.
It starts from a hand-written minimax player and trains a neural network to replace the heuristic at the leaves of the search. Then it plays against itself to make new data and trains the next generation.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)

## How it works

1. **Minimax baseline.** A depth-limited minimax search scores the board with a hand-written heuristic that counts open 2-, 3- and 4-in-a-row windows.
2. **Self-play data.** Minimax plays thousands of games against itself. Each game starts with a few random moves so the games don't all repeat. Every position is saved with the final result.
3. **Value network.** A small CNN (three 3×3 conv layers) learns to predict who wins from a position. Stacked 3×3 layers see a 7×7 area so they can cover any line of four.
4. **Generations.** The trained network plugs into the same search and plays new games. Each generation trains on that data and is checked against the previous best before it replaces it.

`models/` holds every generation (`gen0.pt` to `gen5c.pt`) and `best.pt` is the current champion.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Play against the AI:

```bash
python play_vs_minimax.py   # vs the hand-written heuristic
python play_vs_nn.py        # vs the trained network
```

## Training pipeline

```bash
# 1. Generate games (uses every CPU core but one)
python main.py selfplay --player minimax -n 10000 --depth 3

# 2. Train a network on them
python train.py --model-out models/gen1.pt

# 3. Generate the next generation's data with the network
python main.py selfplay --player neural --model models/gen1.pt -n 10000 --out data/play_001.jsonl

# 4. Compare two players head to head
python evaluate.py --p1 trained --p2 heuristic -n 500
```

Helper scripts:

| Script | What it does |
|---|---|
| `stats.py` | Summarise recorded game files (results and position counts) |
| `dedupe.py` | Remove repeated games so they don't over-weight training |
| `benchmark_depth.py` | Time one minimax move at each search depth |

The game logs are several GB so they are not in the repo. Run the selfplay step to make your own.

## Project layout

```
connect4/
  board.py       board state and win detection (no dependencies)
  heuristics.py  hand-written position score
  players.py     human, random, minimax and neural-minimax players
  network.py     dense and convolutional value networks
  recorder.py    game logging to JSONL / NumPy
  game.py        game loop
main.py          CLI: play / selfplay / export-npz
train.py         train a value network
evaluate.py      head-to-head matches between players
tests/           pytest suite
```

## GPU (optional)

`requirements-directml.txt` sets up a separate DirectML environment for AMD GPUs on Windows.
The model is small so the CPU build is usually enough.

```bash
python -m pytest   # run the tests
```
