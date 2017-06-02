import numpy as np
from keras.models import load_model
import keras
import time


MOVES_CNT = 225
EPS = 5
USED_MOVES = 5
BOARD_SIZE = 15
REWARD = 10
TIMEOUT = 3.5
MAX_DEPTH = 10

model = load_model('TrainedWithRotations')

def make_board(state):
    board = np.zeros((15, 15, 3))
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            board[y][x][state[y][x]] = 1
    return board


def get_player(state):
    cnt = [0, 0, 0]
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            cnt[state[y][x]] += 1
    return 1 if cnt[1] == cnt[2] else 2

def calc_sequence(state, player, x, y, dx, dy):
    cnt = 0
    x += dx
    y += dy
    while (x >= 0 and y >= 0 and x < BOARD_SIZE and y < BOARD_SIZE and state[y][x] == player):
        cnt += 1
        x += dx
        y += dy
    # print('calc %d' % cnt)
    return cnt

def judge(state, player, x, y):
    cnt = calc_sequence(state, player, x, y, -1, 0) + calc_sequence(state, player, x, y, 1, 0) + 1
    if cnt >= 5:
        return 1
    cnt = calc_sequence(state, player, x, y, 0, -1) + calc_sequence(state, player, x, y, 0, 1) + 1
    if cnt >= 5:
        return 1
    cnt = calc_sequence(state, player, x, y, -1, -1) + calc_sequence(state, player, x, y, 1, 1) + 1
    if cnt >= 5:
        return 1
    cnt = calc_sequence(state, player, x, y, -1, 1) + calc_sequence(state, player, x, y, 1, -1) + 1
    if cnt >= 5:
        return 1
    return None

def heuristics_check(state, player, x, y, dx, dy):
    (c1, c2) = (calc_sequence(state, player, x, y, dx, dy), calc_sequence(state, player, x, y, -dx, -dy))
    if c1 + c2 != 3:
        return False
    free = 0
    x1 = x + dx * (c1 + 1)
    y1 = y + dy * (c1 + 1)
    if x1 >= 0 and x1 < BOARD_SIZE and y1 >= 0 and y1 < BOARD_SIZE:
        free += state[y1][x1] == 0

    x2 = x - dx * (c2 + 1)
    y2 = y - dy * (c2 + 1)
    if x2 >= 0 and x2 < BOARD_SIZE and y2 >= 0 and y2 < BOARD_SIZE:
        free += state[y2][x2] == 0

    if free != 2:
        return None
    
    ans = [x + y * BOARD_SIZE]
    if c1 == 0:
        ans.append(x2 + y2 * BOARD_SIZE)

    if c2 == 0:
        ans.append(x1 + y1 * BOARD_SIZE)

    return ans


def heuristics_5(state, player):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if state[y][x] == 0 and judge(state, player, x, y):
                return [x + y * BOARD_SIZE]
    return None


def heuristics_4(state, player):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if state[y][x] == 0:
                ans = (heuristics_check(state, player, x, y, 1, 0)
                        or heuristics_check(state, player, x, y, 0, 1)
                        or heuristics_check(state, player, x, y, 1, 1)
                        or heuristics_check(state, player, x, y, -1, 1))
                if ans:
                    return ans    
    return None


class Node:
    def __init__(self, state):
        self.t = 2
        self.state = state
        self.cur_player = get_player(state)
        self.used = 0
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if state[y][x] != 0:
                    self.used += 1
        
        moves = heuristics_5(state, self.cur_player)
        if not moves:
            moves = heuristics_5(state, 3 - self.cur_player)
        if not moves:
            moves = heuristics_4(state, self.cur_player)
        if not moves:
            moves = heuristics_4(state, 3 - self.cur_player)

        moves_cnt = USED_MOVES
        heuristics = False
        if moves:
            heuristics = True
            self.action_to_move = np.array(moves)
            moves_cnt = len(moves)

        self.cnt = np.ones(moves_cnt, dtype=np.int)
        self.q = np.zeros(moves_cnt)
        self.sum = np.zeros(moves_cnt)
        self.refs = [-1] * moves_cnt

        if heuristics:
            return
        
        pr = model.predict(np.array([make_board(state)]), 1, verbose=0)[0]
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if state[y][x] != 0:
                    pr[y * 15 + x] = -1
        
        self.action_to_move = np.argsort(pr)[MOVES_CNT::-1][:USED_MOVES]

    def chose(self):
        if self.cnt[0] == 1:
            return self.action_to_move[0]
        potential = self.q + EPS * np.sqrt(2 * np.log(self.t) / self.cnt)
        best_actions = np.flatnonzero(potential == potential.max())
        return self.action_to_move[np.random.choice(best_actions)]

    def update(self, move, result):
        action = np.where(self.action_to_move == move)[0][0]
        self.t += 1
        self.cnt[action] += 1
        self.sum[action] += result
        self.q[action] = self.sum[action] / self.cnt[action]

    def get_ref(self, move):
        action = np.where(self.action_to_move == move)[0][0]
        if self.refs[action] == -1:
            y = move // 15
            x = move % 15
            state = self.state.copy()
            state[y][x] = self.cur_player
            self.refs[action] = Node(state)
        return self.refs[action]

    def judge(self, move):
        y = move // 15
        x = move % 15
        if self.used == BOARD_SIZE * BOARD_SIZE - 1:
            return 0
        return judge(self.state, self.cur_player, x, y)

    def get_best(self):
        action = np.argmax(self.q)
        return self.action_to_move[action]

def calc_used(state):
    cnt = 0
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if state[y][x]:
                cnt += 1
    return cnt

class MST:
    def __init__(self, state):
        self.root = Node(state)

    def get_move(self, iters=1, timeout=TIMEOUT):
        time_start = time.time()

        if len(self.root.action_to_move) == 1:
            return self.root.chose()

        for i in range(iters):
            self.calc(self.root)
            if time.time() - time_start > timeout:
                break

        return self.root.get_best()

    def calc(self, cur_node, depth=0):
        if depth == MAX_DEPTH:
            return 0
        move = cur_node.chose() if depth < 4 else cur_node.action_to_move[0]
        status = cur_node.judge(move)
        if status == None:
            status = self.calc(cur_node.get_ref(move), depth=depth + 1)
        cur_node.update(move, status * REWARD)

        if abs(status) > 1:
            return -status * 0.8
        return -status
