from .board import Board, PLAYER_ONE, PLAYER_TWO, other_player


def evaluate_window(window: list[int], player: int, opponent: int) -> float:
    """Score a single group of 4 cells for `player`."""
    player_count = window.count(player)
    opp_count = window.count(opponent)
    empty_count = window.count(0)

    # TODO: your scoring rules go here.
    # Think through these cases, in this order:
    #   1. player_count == 4          -> player already won this window
    #   2. opp_count == 4             -> opponent already won this window
    #   3. player_count > 0 and opp_count > 0
    #                                  -> mixed window, nobody can complete it
    #   4. player_count == 3 and empty_count == 1
    #                                  -> one move from a win, strong
    #   5. player_count == 2 and empty_count == 2
    #                                  -> weak but positive
    #   6. mirror cases 4 and 5 for opp_count, as NEGATIVE numbers
    #   7. anything else               -> 0

    if (player_count > 0 and opp_count > 0):
        return 0.0
    if (player_count == 4):
        return 100.0
    if (player_count == 3 and empty_count == 1):
        return 10.0
    if (player_count == 2 and empty_count == 2):
        return 1.0
    if (opp_count == 4):
        return -100.0
    if (opp_count == 3 and empty_count == 1):
        return -10.0
    if (opp_count == 2 and empty_count == 2):
        return -1.0
    
    return 0.0

def score_position(board: Board, player: int) -> float:
    """Score the whole board for `player`, no lookahead."""
    opponent = other_player(player)
    total = 0.0

    # Vertical check
    for row in range(board.rows):
        for col in range(board.cols - 3):
            window = [board.grid[row][i] for i in range(col, col+4)]
            total += evaluate_window(window, player, opponent)

    # Horizontal check
    for row in range(board.rows - 3):
            for col in range(board.cols):
                window = [board.grid[i][col] for i in range(row, row+4)]
                total += evaluate_window(window, player, opponent)

    # Diagonal check
    for row in range(board.rows - 3):
        for col in range(board.cols - 3):
            window = [board.grid[row+i][col+i] for i in range(4)]
            total += evaluate_window(window, player, opponent)
    for row in range(3, board.rows):
        for col in range(board.cols - 3):
            window = [board.grid[row-i][col+i] for i in range(4)]
            total += evaluate_window(window, player, opponent)

    center_col = board.cols//2
    window = [board.grid[row][center_col] for row in range(board.rows)]
    for i in range(len(window)):
        if (window[i] == player): total += 0.2
        elif (window[i] == opponent): total -= 0.2
        
    return total