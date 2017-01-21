from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import Adam, Adadelta
from keras import backend as K
from keras.regularizers import l2

np.random.seed(1337)

batch_size = 500
nb_classes = 500
nb_epoch = 2
pool_size = (2, 2)
pool_size2 = (3, 3)
input_shape = (40, 40, 1)
init = "he_uniform"

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', input_shape=input_shape, name='conv1'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool1'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool2'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='conv3'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='conv4'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool3'))
model.add(BatchNormalization())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(500, activation='softmax', init=init, name='dense_last'))

opt = Adadelta(lr=0.3, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# model.load_weights('solution6')

(x_test, y_test) = np.load('data/x_test.npy'), np.load('data/y_test.npy')
(x_train, y_train) = np.load('data/x_train.npy'), np.load('data/y_train.npy')

X_train = x_train.astype('float16')
X_test = x_test.astype('float16')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1, validation_data=(X_test, Y_test))
model.save_weights('solution6')
model.save('Model')

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
