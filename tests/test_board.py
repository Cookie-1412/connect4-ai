import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from connect4.board import Board, PLAYER_ONE, PLAYER_TWO


def test_initial_state():
    b = Board()
    assert b.current_player == PLAYER_ONE
    assert b.winner is None
    assert len(b.legal_moves()) == 7


def test_drop_piece_lands_at_bottom():
    b = Board()
    row = b.drop_piece(3)
    assert row == 5
    assert b.grid[5][3] == PLAYER_ONE
    assert b.current_player == PLAYER_TWO


def test_column_full_raises():
    b = Board()
    # Filling one column alone alternates P1/P2 turn by turn, so it never
    # produces 4-in-a-row vertically before the column fills up.
    for _ in range(6):
        b.drop_piece(0)
    assert all(b.grid[r][0] != 0 for r in range(6))
    assert b.winner is None
    try:
        b.drop_piece(0)
        assert False, "expected ValueError for full column"
    except ValueError:
        pass


def test_horizontal_win():
    b = Board()
    # P1 plays 0,1,2,3 on bottom row; P2 plays elsewhere
    moves = [0, 0, 1, 1, 2, 2, 3]
    for col in moves:
        b.drop_piece(col)
    assert b.winner == PLAYER_ONE


def test_vertical_win():
    b = Board()
    moves = [0, 1, 0, 1, 0, 1, 0]
    for col in moves:
        b.drop_piece(col)
    assert b.winner == PLAYER_ONE


def test_diagonal_win():
    b = Board()
    # Builds a falling diagonal for P1 at (5,0),(4,1),(3,2),(2,3),
    # using col 6 as junk moves so P2 can stack the needed fillers.
    moves = [0, 1, 1, 2, 6, 2, 2, 3, 6, 3, 6, 3, 3]
    for col in moves:
        b.drop_piece(col)
    assert b.winner == PLAYER_ONE
    assert b.grid[2][3] == PLAYER_ONE
    assert b.grid[3][2] == PLAYER_ONE
    assert b.grid[4][1] == PLAYER_ONE
    assert b.grid[5][0] == PLAYER_ONE


def test_draw_detection():
    b = Board(cols=1, rows=1)
    b.drop_piece(0)
    assert b.is_game_over()
    assert b.is_draw() or b.winner is not None


def test_clone_is_independent():
    b = Board()
    b.drop_piece(0)
    c = b.clone()
    c.drop_piece(1)
    assert b.grid != c.grid
    assert b.current_player != c.current_player or b.move_count != c.move_count
