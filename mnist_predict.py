#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: TaoJunXin
@contact: taojunxin@xinktech.com
@file: mnist_predict.py
@time: 2018/12/5 17:35
@desc:

"""

import cv2
import numpy as np
from keras.models import load_model

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))
# sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':0}))

img = cv2.imread('2.png', 0)

img2 = np.reshape(img, (1, 28, 28, 1))

model = load_model('mnist.h5')
# print(model.summary())

img2 = img2.astype('float32')
img2 /= 255

r = model.predict(img2)
print(r)
_max = np.argmax(r[0])
print(_max)
# print(img2.shape)
