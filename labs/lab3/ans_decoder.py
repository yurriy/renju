import numpy as np
from scipy.misc import toimage, imresize

indexes = dict()
indexes_inv = dict()

def calc_indexes(data):
    labels_uniques = np.unique(data[:, 1])
    labels_uniques.sort()
    for i in range(labels_uniques.size):
        indexes[labels_uniques[i]] = i
        indexes_inv[i] = labels_uniques[i]

data = np.load('train.npy')
calc_indexes(data)
ans = np.load('ans.npy')

f = open('submission.csv', 'w')
f.write('Id,Category\n')
for i in range(len(ans)):
    ans[i] = indexes_inv[ans[i]]
    f.write(str(i + 1) + ',' + str(ans[i]) + '\n')

f.close()
