"""Player interfaces. Any AI trained later just needs to implement
`get_move(board) -> int` and can drop straight into `play_game`."""

from __future__ import annotations

import random, math, torch
from abc import ABC, abstractmethod

from connect4.heuristics import score_position
from connect4.network import board_to_tensor

from .board import Board
from .network import ValueNetwork

# What _minimax returns for a decided position. Sampling must never pick
# away from a score this large -- that would throw away a forced win.
TERMINAL_SCORE = 1000.0


class BasePlayer(ABC):
    name: str = "player"

    @abstractmethod
    def get_move(self, board: Board) -> int:
        """Return a legal column index given the current board state."""
        raise NotImplementedError


class HumanPlayer(BasePlayer):
    name = "human"

    def get_move(self, board: Board) -> int:
        legal = board.legal_moves()
        while True:
            raw = input(f"Player {board.current_player}, choose a column {legal}: ")
            try:
                col = int(raw)
            except ValueError:
                print("Enter a number.")
                continue
            if col not in legal:
                print("That column is not available.")
                continue
            return col


class RandomPlayer(BasePlayer):
    """Picks uniformly among legal moves. Useful as a baseline opponent
    and for generating self-play training data cheaply."""

    name = "random"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def get_move(self, board: Board) -> int:
        return self.rng.choice(board.legal_moves())


class MinimaxPlayer(BasePlayer):
    name = "minimax"

    def __init__(self, depth: int = 3):     # depth = prediction for amount of moves
        self.depth = depth

    def get_move(self, board: Board) -> int:
        # TODO: loop over board.legal_moves(), simulate each with
        # board.clone() + drop_piece(col), call self._minimax(...) on
        # the resulting position, and remember which column gave the
        # best score. Return that column.
        
        max_score = -math.inf
        return_col = 0
        for col in board.legal_moves():
            clone = board.clone()
            clone.drop_piece(col)
            score = self._minimax(clone, self.depth - 1, False, board.current_player)
            if (score > max_score):
                max_score = score
                return_col = col

        return return_col

    def _minimax(self, board: Board, depth: int, maximizing: bool, perspective_player: int) -> float:
        # Base cases
        if (board.winner):
            if (board.winner == perspective_player): return 1000.0          # Win
            return -1000.0                                                  # Lose
        if (board.legal_moves() == []): return 0.0                          # Draw
        if (depth <= 0): return score_position(board, perspective_player)   # Out of budget

        # Recursion
        max_score = -math.inf
        min_score = math.inf
        for col in board.legal_moves():
            clone = board.clone()
            clone.drop_piece(col)
            score = self._minimax(clone, depth - 1, not maximizing, perspective_player)
            if (maximizing and score > max_score):
                max_score = score
            if (not maximizing and score < min_score):
                min_score = score
        
        if (maximizing): return max_score
        return min_score

class NeuralMinimaxPlayer(BasePlayer):
    name = "neural-minimax"

    def __init__(self, network, depth: int = 3, epsilon: float = 0.0,
                 sample_plies: int = 0, rng: random.Random | None = None):
        """epsilon / sample_plies turn on exploration for self-play data
        generation. With the defaults this plays exactly as before: strongest
        move, every time. That matters -- evaluation and human games must stay
        deterministic, only data generation wants variety.

        rng is passed in rather than created here for the same reason
        RandomPlayer takes one: every worker process must seed independently
        or they all generate identical games."""
        self.network = network
        self.depth = depth
        self.epsilon = epsilon
        self.sample_plies = sample_plies
        self.rng = rng or random.Random()

    def get_move(self, board: Board) -> int:
        return self._choose(board, self._score_moves(board))

    def _score_moves(self, board: Board):
        """Every legal move paired with its search score. Kept separate from
        the choice so exploration can see the whole picture -- a wrapper that
        only sees the chosen column can't tell a close second from a blunder."""
        scored = []
        for col in board.legal_moves():
            clone = board.clone()
            clone.drop_piece(col)
            scored.append(
                (col, self._minimax(clone, self.depth - 1, False, board.current_player)))
        return scored

    def _choose(self, board: Board, scored) -> int:
        """Pick the best move, or explore among near-equal ones."""
        best_col, best_score = max(scored, key=lambda pair: pair[1])

        # Play it straight when exploration is off, once the opening is over,
        # or whenever the position is already decided. Randomising a forced
        # win throws away the game; randomising late corrupts the outcome
        # label, which is the whole point of recording the game.
        if (self.epsilon <= 0
                or board.move_count >= self.sample_plies
                or abs(best_score) >= TERMINAL_SCORE):
            return best_col

        near_best = [col for col, score in scored if score >= best_score - self.epsilon]
        return self.rng.choice(near_best)

    def _minimax(self, board, depth, maximizing, perspective_player) -> float:
        if (board.winner):
            if (board.winner == perspective_player): return TERMINAL_SCORE  # Win
            return -TERMINAL_SCORE                                          # Lose
        if (board.legal_moves() == []): return 0.0                          # Draw
        if (depth <= 0):                                                    # RL
            # The network was trained on positions encoded from the SIDE TO
            # MOVE's point of view -- every training tensor put the player
            # about to move in plane 0. Encoding from perspective_player here
            # would hand it a board it never saw during training on every
            # other ply. So ask it the question it was trained to answer,
            # then flip the sign to express the answer as perspective_player.
            mover = board.current_player
            tensor = board_to_tensor(board, mover).unsqueeze(0)
            with torch.no_grad():
                value = self.network(tensor).item()
            return value if mover == perspective_player else -value


        # Recursion
        max_score = -math.inf
        min_score = math.inf
        for col in board.legal_moves():
            clone = board.clone()
            clone.drop_piece(col)
            score = self._minimax(clone, depth - 1, not maximizing, perspective_player)
            if (maximizing and score > max_score):
                max_score = score
            if (not maximizing and score < min_score):
                min_score = score
        
        if (maximizing): return max_score
        return min_score

class RandomOpeningPlayer(BasePlayer):
    name = "random-opening"

    def __init__(self, inner: BasePlayer, random_moves: int = 6, rng: random.Random | None = None):
        self.inner = inner
        self.random_moves = random_moves
        self.rng = rng or random.Random()
        # Surface what's being wrapped, so recorded games are traceable.
        self.name = f"random-opening({inner.name})"

    def get_move(self, board: Board) -> int:
        if (board.move_count < self.random_moves): return self.rng.choice(board.legal_moves())
        return self.inner.get_move(board)