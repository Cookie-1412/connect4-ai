"""Records games in a format aimed at later AI training.

Every finished game is appended as one JSON line to a JSONL file. Each
line contains the full move-by-move trajectory plus, for every move, a
`value_target` from the perspective of the player who moved: +1 if that
player eventually won, -1 if they lost, 0 for a draw. That is the
standard target used for training a value network (e.g. AlphaZero-style)
or for computing policy-gradient returns, so no extra preprocessing is
needed to start training later.

JSONL was chosen over a single JSON array because it lets self-play runs
append incrementally (crash-safe, streamable) and lets training code
read the file lazily line-by-line without loading everything into memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .board import Board, PLAYER_ONE, PLAYER_TWO


@dataclass
class MoveRecord:
    move_index: int
    player: int
    board_before: List[List[int]]
    action: int
    board_after: List[List[int]]
    value_target: Optional[float] = None  # filled in once the game ends


@dataclass
class GameRecord:
    game_id: str
    player_names: List[str]
    moves: List[MoveRecord] = field(default_factory=list)
    winner: Optional[int] = None
    outcome: str = "in_progress"  # "win" | "draw"

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "player_names": self.player_names,
            "winner": self.winner,
            "outcome": self.outcome,
            "num_moves": len(self.moves),
            "moves": [
                {
                    "move_index": m.move_index,
                    "player": m.player,
                    "board_before": m.board_before,
                    "action": m.action,
                    "board_after": m.board_after,
                    "value_target": m.value_target,
                }
                for m in self.moves
            ],
        }


class GameRecorder:
    """Attach to a single game: call `record_move` after every move, then
    `finish` once the game is over. Call `save` (or use as a context
    manager over multiple games) to append to a JSONL file."""

    def __init__(self, player_names: List[str], game_id: str):
        self.record = GameRecord(game_id=game_id, player_names=player_names)

    def record_move(self, board_before: Board, action: int, board_after: Board) -> None:
        move = MoveRecord(
            move_index=len(self.record.moves),
            player=board_before.current_player,
            board_before=board_before.to_array(),
            action=action,
            board_after=board_after.to_array(),
        )
        self.record.moves.append(move)

    def finish(self, board: Board) -> GameRecord:
        self.record.winner = board.winner
        self.record.outcome = "draw" if board.winner is None else "win"

        for move in self.record.moves:
            if board.winner is None:
                move.value_target = 0.0
            elif move.player == board.winner:
                move.value_target = 1.0
            else:
                move.value_target = -1.0

        return self.record

    def append_to_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self.record.to_dict()) + "\n")


def load_games(path: str | Path):
    """Lazily yield game dicts from a JSONL file written by GameRecorder."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def export_to_numpy(jsonl_path: str | Path, out_path: str | Path) -> None:
    """Flatten every move across every game into stacked arrays and save
    as a single .npz: boards (N, rows, cols), actions (N,), values (N,),
    players (N,). Ready to feed into a PyTorch/TensorFlow Dataset.

    Requires numpy (not a hard dependency of the core package).
    """
    import numpy as np

    boards, actions, values, players = [], [], [], []
    for game in load_games(jsonl_path):
        for move in game["moves"]:
            boards.append(move["board_before"])
            actions.append(move["action"])
            values.append(move["value_target"])
            players.append(move["player"])

    np.savez_compressed(
        out_path,
        boards=np.array(boards, dtype=np.int8),
        actions=np.array(actions, dtype=np.int8),
        values=np.array(values, dtype=np.float32),
        players=np.array(players, dtype=np.int8),
    )
