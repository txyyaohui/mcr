import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

'''Trains a simple convnet on the MNIST dataset using maximal correlation regression : https://ieeexplore.ieee.org/abstract/document/8979352
Xiangxiang Xu <xiangxiangxu.thu@gmail.com>

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).

Log-loss v.s. H-score

V1: 2018-06-06 10:13:12
V2: 2018-11-19 16:07:08
V3: 2018-11-29 21:12:23
V4: 2018-12-03 21:33:10
V5: 2018-12-22 10:26:27
V6: 2019-05-18 22:13:05 No need to import tf.
V7: 2019-05-19 23:39:08 New functions
'''
import os

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Lambda
from keras import backend as K
import numpy as np

from func import *


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

def mnist_load_data():
    test_path = '/home/feng/.keras/datasets/mnist.npz'
    if os.path.exists(test_path):
        results = mnist.load_data(test_path)
    else:
        results = mnist.load_data()
    return results
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist_load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

fdim = 128
gdim = fdim


# Log-loss
input_x = Input(shape = input_shape)
f_log = feature_f(input_x, fdim)
predictions = Dense(num_classes, activation='softmax')(f_log)
model_log = Model(inputs=input_x, outputs=predictions)

model_log.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print('# of Paras', model_log.count_params())
model_log.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))
acc_log = model_log.evaluate(x_test, y_test, verbose=0)[1]
print('acc_log = ', acc_log)
model_log_f = Model(inputs = model_log.input, outputs = model_log.layers[-1].input)
f_test_log = model_log_f.predict(x_test)


# H-score
input_x = Input(shape = input_shape)
f = feature_f(input_x, fdim)

input_y = Input(shape = (num_classes, ))
g = Dense(gdim)(input_y) # g should not be activated.
# g = Dropout(0.5)(g) # no dropout

loss = Lambda(neg_hscore)([f, g])
model = Model(inputs = [input_x, input_y], outputs = loss)
model.compile(optimizer=keras.optimizers.Adadelta(), loss = lambda y_true,y_pred: y_pred)
model.fit([x_train, y_train],
          np.zeros([y_train.shape[0], 1]),
          batch_size = batch_size,
          epochs = epochs,
          validation_data=([x_test, y_test], np.zeros([y_test.shape[0], 1])))#validation_split = 0.2)
model_f = Model(inputs = input_x, outputs = f)
model_g = Model(inputs = input_y, outputs = g)
f_test = model_f.predict(x_test)
f_test0 = f_test - np.mean(f_test, axis = 0)
g_val = model_g.predict(np.eye(10))
py = np.mean(y_train, axis = 0)
g_val0 = g_val - np.matmul(py, g_val)  # get zero-mean g(Y)
# BUG fixed
pygx = py * (1 + np.matmul(f_test0, g_val0.T))
acc = np.mean(np.argmax(pygx, axis = 1) == np.argmax(y_test, axis = 1))
print(acc)
