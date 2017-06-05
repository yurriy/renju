import asyncio
import websockets
import numpy as np
import os
from mst import MST


TIMEOUT = 3
BOARD_SIZE = 15

clients_cnt = 0
games_cnt = 0


def get_move(board):
    mst = MST(board)
    return mst.get_move(iters=500, timeout=TIMEOUT)


def fix_format(s):
    if s.count('1') >= s.count('2'):
        return s
    s = s.replace('1', '3')
    s = s.replace('2', '1')
    return s.replace('3', '2')


async def play(websocket, log_file):
    global clients_cnt
    try:
        mode = await websocket.recv()
        clients_cnt += 1
        print("client %d connected" % clients_cnt)
        while True:
            print("%d CLIENTS NOW" % clients_cnt)
            message = await websocket.recv()
            message = fix_format(message)
            
            board = np.array(list(map(lambda x: 0 if x == '' else int(x), message.split(','))))
            board = np.hstack((board, np.zeros(225 - len(board), dtype=int))).reshape((BOARD_SIZE, BOARD_SIZE))
            log_file.write(str(board) + '\n')

            move = get_move(board)
            log_file.write(str((move // BOARD_SIZE, move % BOARD_SIZE)) + '\n')

            await websocket.send(str(move))
    except:
        clients_cnt -= 1
        print("client disconnected")
        print("%d CLIENTS NOW" % clients_cnt)


async def handle(websocket, path):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    global games_cnt
    with open('logs/game_%d.txt' % games_cnt, 'w') as log_file:
        games_cnt += 1
        await play(websocket, log_file)


start_server = websockets.serve(handle, '127.0.0.1', 12347)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
