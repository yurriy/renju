# this script trains policy network, which predicts next move by board, on renju games dataset

from __future__ import print_function
import keras
import os
import random
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, Conv3D, MaxPooling2D, SeparableConv2D
from keras import backend as K
from keras import regularizers


data = []

with open('data.txt') as f:
    cnt = 0
    game = f.readline()
    ans = f.readline()
    while (game != '') and (ans != ''):

        if (game != os.linesep) and (ans != os.linesep):
            data.append((game, ans))
            cnt += 1
        
        if cnt % 100000 == 0:
            print(cnt)
        
        game = f.readline()
        ans = f.readline()

    print('games read: ', cnt)


# rotation[k][(y, x)] stores the transformation of (y, x) after k rotations of board by 90 deg
rotation = np.empty((4, 15, 15, 2), dtype=np.int)

def rotations_precalc():
    for i in range(225):
        (y, x) = (i // 15, i % 15)
        board = np.zeros((15, 15))
        board[y][x] = 1
        for k in range(4):
            rotated_board = np.rot90(board, k)
            position = np.argmax(rotated_board)
            rotation[k][(y, x)] = (position // 15, position % 15)

rotations_precalc()


def random_transform(board, min_x, max_x, min_y, max_y, ans_move):
    (ans_y, ans_x) = (ans_move // 15, ans_move % 15)
    
    (min_y, min_x) = (min(min_y, ans_y), min(min_x, ans_x))
    (max_y, max_x) = (max(max_y, ans_y), max(max_x, ans_x))

    rotations_cnt = np.random.randint(0, 4)
    board = np.rot90(board, rotations_cnt)
    (ans_y, ans_x) = rotation[rotations_cnt][(ans_y, ans_x)]
    for i in range(rotations_cnt):
        (min_y, min_x, max_y, max_x) = (14 - max_x, min_y, 14 - min_x, max_y)

    if np.random.rand() > 0.5:
        board = np.fliplr(board)
        (min_x, max_x) = (14 - max_x, 14 - min_x)
        ans_x = 14 - ans_x

    x_shift = np.random.randint(-min_x, 15 - max_x)
    y_shift = np.random.randint(-min_y, 15 - max_y)

    return (np.roll(board, (x_shift, y_shift), axis=(1, 0)), ans_x + x_shift + 15 * (ans_y + y_shift))


def generator(data, batch_size):
    n = len(data)
    cur_game = 0
    cur_move = -1
            
    while True:

        batch_boards = []
        batch_moves = []
        (min_x, min_y), (max_x, max_y) = (15, 15), (0, 0)

        filled = 0

        while (filled < batch_size):

            if cur_move == -1 or cur_move >= len(ans_moves):
                cur_game = (cur_game + 1) % n
                cur_move = 0
                moves = list(map(int, data[cur_game][0].split(' ')))
                ans_moves = list(map(int, data[cur_game][1].split(' ')))
                (min_x, min_y), (max_x, max_y) = (15, 15), (0, 0)
                board1 = np.dstack((np.ones((15, 15, 1)), np.zeros((15, 15, 2))))
                board2 = board1.copy()

            x = (moves[cur_move] - 1) % 15
            y = (moves[cur_move] - 1) // 15
            
            (min_y, min_x) = (min(min_y, y), min(min_x, x))
            (max_y, max_x) = (max(max_y, y), max(max_x, x))

            board1[y][x][1 + cur_move % 2] = 1
            board2[y][x][2 - cur_move % 2] = 1
            board1[y][x][0] = 0
            board2[y][x][0] = 0

            board = board1 if cur_move % 2 else board2
            ans_move = ans_moves[cur_move] - 1

            sample = random_transform(board, min_x, max_x, min_y, max_y, ans_move)
            batch_boards.append(sample[0])
            batch_moves.append(sample[1])

            filled += 1
            cur_move += 1

        yield np.array(batch_boards), keras.utils.to_categorical(batch_moves, 225)


batch_size = 128
num_classes = 225
epochs = 100
steps_per_epoch = 10000

gen = generator(data, batch_size)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(15, 15, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adamax(),
              metrics=['accuracy'])

model.load_weights('TrainedWithRotations')

save_callback = keras.callbacks.ModelCheckpoint('TrainedWithRotations',
    monitor='acc',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1)

model.fit_generator(gen,
    steps_per_epoch,
    epochs=epochs,
    verbose=1,
    callbacks=[save_callback],
    validation_data=None,
    validation_steps=None,
    class_weight=None,
    max_q_size=10,
    workers=1,
    pickle_safe=False,
    initial_epoch=33)
