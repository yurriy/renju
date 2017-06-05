import numpy as np
import cv2
from scipy.misc import toimage, imresize

n = 40

(im_h, im_w) = (n, n)

def invert_colors(im):
    return np.abs(im - 255)

data = np.load('test.npy')
cnt = len(data)
prev_len = 0
x_full = np.zeros((cnt, im_h, im_w))
for i in range(cnt):
    x_full[i] = imresize(data[i], size=(im_h, im_w))
    x_full[i] = np.abs(x_full[i] - 255)
    progress = str(i / len(data) * 100)[:4] + '%'
    print('\r' * prev_len, progress, end="")
    prev_len = len(progress)

print("\nFormatting has finished")

x_full = x_full.reshape(cnt, im_h, im_w, 1)

np.save('data/test_data', x_full)