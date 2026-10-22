"""Play against the TRAINED neural network.

Run:  python play_vs_nn.py
"""

from pathlib import Path

import torch

from connect4.game import play_game
from connect4.network import load_network
from connect4.players import HumanPlayer, NeuralMinimaxPlayer

MODEL_FILE = Path("data/value_net.pt")
DEPTH = 3

if not MODEL_FILE.exists():
    raise SystemExit(f"No trained model at {MODEL_FILE}. Run `python train.py` first.")

net = load_network(MODEL_FILE)

print(f"Playing against the trained network (search depth {DEPTH}). You are X.\n")
play_game(HumanPlayer(), NeuralMinimaxPlayer(net, depth=DEPTH), verbose=True)