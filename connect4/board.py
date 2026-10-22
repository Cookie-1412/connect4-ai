"""Pure Connect 4 game engine: no I/O, no printing, no randomness.

Kept dependency-free and side-effect-free on purpose so the exact same
Board class can drive a CLI game, a batch of self-play games for
generating training data, or a future AI agent's search/inference loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

ROWS = 6
COLS = 7
CONNECT = 4

EMPTY = 0
PLAYER_ONE = 1
PLAYER_TWO = 2

Player = int  # PLAYER_ONE or PLAYER_TWO


def other_player(player: Player) -> Player:
    return PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE


@dataclass
class Board:
    rows: int = ROWS
    cols: int = COLS
    grid: List[List[int]] = field(default=None)  # type: ignore[assignment]
    current_player: Player = PLAYER_ONE
    winner: Optional[Player] = None
    last_move_col: Optional[int] = None
    move_count: int = 0

    def __post_init__(self):
        if self.grid is None:
            self.grid = [[EMPTY] * self.cols for _ in range(self.rows)]

    def clone(self) -> "Board":
        return Board(
            rows=self.rows,
            cols=self.cols,
            grid=[row[:] for row in self.grid],
            current_player=self.current_player,
            winner=self.winner,
            last_move_col=self.last_move_col,
            move_count=self.move_count,
        )

    def legal_moves(self) -> List[int]:
        if self.winner is not None:
            return []
        return [c for c in range(self.cols) if self.grid[0][c] == EMPTY]

    def is_full(self) -> bool:
        return all(self.grid[0][c] != EMPTY for c in range(self.cols))

    def is_game_over(self) -> bool:
        return self.winner is not None or self.is_full()

    def is_draw(self) -> bool:
        return self.winner is None and self.is_full()

    def drop_piece(self, col: int) -> int:
        """Drop the current player's piece into `col`. Returns the row it
        landed on. Mutates the board in place and flips current_player
        unless the move ends the game (win/draw), matching how most
        training loops want to detect terminal states immediately."""
        if self.is_game_over():
            raise ValueError("game is already over")
        if col < 0 or col >= self.cols:
            raise ValueError(f"column {col} out of range")
        if self.grid[0][col] != EMPTY:
            raise ValueError(f"column {col} is full")

        row = self.rows - 1
        while self.grid[row][col] != EMPTY:
            row -= 1
        self.grid[row][col] = self.current_player
        self.last_move_col = col
        self.move_count += 1

        if self._check_win(row, col):
            self.winner = self.current_player
        else:
            self.current_player = other_player(self.current_player)

        return row

    def _check_win(self, row: int, col: int) -> bool:
        player = self.grid[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)
            if count >= CONNECT:
                return True
        return False

    def _count_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def to_array(self) -> List[List[int]]:
        """Plain nested-list snapshot of the board, safe to json.dumps
        or feed into numpy.array()/torch.tensor() for training."""
        return [row[:] for row in self.grid]

    def __str__(self) -> str:
        symbols = {EMPTY: ".", PLAYER_ONE: "X", PLAYER_TWO: "O"}
        lines = [" ".join(symbols[cell] for cell in row) for row in self.grid]
        lines.append(" ".join(str(c) for c in range(self.cols)))
        return "\n".join(lines)
