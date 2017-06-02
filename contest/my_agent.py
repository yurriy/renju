import logging
import random

import backend
import renju
import util
import numpy as np
from mst import MST

TIMEOUT = 9

def move_to_str(move):
    y = move // 15
    x = move % 15
    return chr(x + ord('a')) + str(y + 1)

def fix_board(board):
    board[board == 1] = 2
    board[board == -1] = 1
    return board

def main():
    try:
        while True:
            game = backend.wait_for_game_update()

            state = fix_board(game.board().copy())
            mst = MST(state)
            move = mst.get_move(iters=1000, timeout=TIMEOUT)
            ans = move_to_str(move)

            backend.move(ans)
    except:
        pass


if __name__ == "__main__":
    main()
