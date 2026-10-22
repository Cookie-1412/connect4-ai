import torch
import torch.nn as nn
from .board import Board, PLAYER_ONE, PLAYER_TWO, other_player

def board_to_tensor(board: Board, player: int) -> torch.Tensor:
    plane = torch.zeros(2, board.rows, board.cols)
    opponent = other_player(player)

    for row in range(board.rows):
        for col in range(board.cols):
            cell = board.grid[row][col]
            if (cell == 0): continue
            if (cell == player): plane[0, row, col] = 1.0
            elif (cell == opponent): plane[1, row, col] = 1.0
            
    
    return plane

def boards_to_tensor(grids, players) -> torch.Tensor:
    """Bulk equivalent of board_to_tensor: converts MANY positions at once.

    board_to_tensor loops over all 42 cells in Python, which is fine for one
    position inside a search but costs ~460s when applied to 3M training
    positions. numpy does the same comparisons on the whole stack at once,
    so the loop runs in C instead. Output is identical, as uint8 (every
    value is 0 or 1) -- callers convert to float where they need it.

    grids:   sequence of rows x cols int grids (move["board_before"])
    players: matching sequence of "whose point of view" ints
    """
    import numpy as np

    g = np.asarray(grids, dtype=np.int8)                       # (N, rows, cols)
    me = np.asarray(players, dtype=np.int8).reshape(-1, 1, 1)  # (N, 1, 1)
    them = np.where(me == PLAYER_ONE, PLAYER_TWO, PLAYER_ONE)
    planes = np.stack([g == me, g == them], axis=1)            # (N, 2, rows, cols)
    return torch.from_numpy(planes.astype(np.uint8))


class ValueNetwork(nn.Module):
    """Dense network. Sees the board as 84 unrelated numbers, so it has to
    learn every 4-in-a-row window independently. Kept as a baseline."""

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(84, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvValueNetwork(nn.Module):
    """Convolutional network. Each conv layer slides a 3x3 window across the
    board, so a pattern like "three of mine in a row" is learned ONCE and
    recognized everywhere, instead of separately per board location.

    padding=1 keeps the board 6x7 through every conv layer; three stacked
    3x3 layers give each output cell a 7x7 view, enough to cover any
    4-in-a-row line (horizontal, vertical, or diagonal)."""

    def __init__(self, channels: int = 64):
        super().__init__()

        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(channels * 6 * 7, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


ARCHITECTURES = {
    "dense": ValueNetwork,
    "conv": ConvValueNetwork,
}


def save_network(net: nn.Module, path) -> None:
    """Save weights plus which architecture they belong to, so loading
    can't silently mismatch. Weights are moved to CPU first: a state_dict
    saved straight off the DirectML device won't load with weights_only=True."""
    arch = next(name for name, cls in ARCHITECTURES.items() if isinstance(net, cls))
    torch.save(
        {"arch": arch, "state_dict": {k: v.cpu() for k, v in net.state_dict().items()}},
        path,
    )


def load_network(path) -> nn.Module:
    """Rebuild the right architecture from a checkpoint and load its weights."""
    blob = torch.load(path, map_location="cpu", weights_only=True)

    if "arch" in blob:
        arch, state = blob["arch"], blob["state_dict"]
    else:
        # Older checkpoints were a bare state_dict with no architecture recorded.
        arch, state = "dense", blob

    net = ARCHITECTURES[arch]()
    net.load_state_dict(state)
    net.eval()
    return net