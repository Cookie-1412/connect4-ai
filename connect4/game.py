from __future__ import annotations

import uuid
from typing import Optional

from .board import Board, PLAYER_ONE, PLAYER_TWO
from .players import BasePlayer
from .recorder import GameRecorder


def play_game(
    player_one: BasePlayer,
    player_two: BasePlayer,
    verbose: bool = False,
    record: bool = False,
    game_id: Optional[str] = None,
) -> tuple[Board, Optional[GameRecorder]]:
    """Run one game to completion. Returns the final board and, if
    `record=True`, the GameRecorder holding the full trajectory."""
    board = Board()
    players = {PLAYER_ONE: player_one, PLAYER_TWO: player_two}
    recorder = None
    if record:
        recorder = GameRecorder(
            player_names=[player_one.name, player_two.name],
            game_id=game_id or str(uuid.uuid4()),
        )

    if verbose:
        print(board)

    while not board.is_game_over():
        current = players[board.current_player]
        col = current.get_move(board)

        board_before = board.clone() if recorder else None
        board.drop_piece(col)

        if recorder:
            recorder.record_move(board_before, col, board)

        if verbose:
            print()
            print(board)

    if recorder:
        recorder.finish(board)

    if verbose:
        if board.winner:
            print(f"\nPlayer {board.winner} ({players[board.winner].name}) wins!")
        else:
            print("\nIt's a draw!")

    return board, recorder
