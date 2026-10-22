import time

from connect4.board import Board
from connect4.players import MinimaxPlayer

# Empty board is close to the worst case: all 7 columns are legal moves,
# so nothing gets cut off early just because a column happens to be full.
board = Board()

for depth in range(1, 8):
    ai = MinimaxPlayer(depth=depth)
    start = time.time()
    move = ai.get_move(board)
    elapsed = time.time() - start
    print(f"depth={depth}: {elapsed:.3f}s (chose column {move})")
