from __future__ import print_function
import numpy as np
import keras

model = keras.models.load_model('Model')

x_test = np.load('data/test_data.npy')
X_test = x_test.astype('float16')
X_test /= 255

print('Calculating answer')
ans = model.predict_classes(X_test)
np.save('ans', ans)
print('Answer is saved')