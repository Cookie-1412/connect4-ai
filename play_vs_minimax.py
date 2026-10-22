from connect4.game import play_game
from connect4.players import HumanPlayer, MinimaxPlayer

play_game(HumanPlayer(), MinimaxPlayer(depth=3), verbose=True)