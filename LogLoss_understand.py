#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:21:30 2019

@author: aramos
"""

from sklearn import metrics, preprocessing
from IPython import display
import tensorflow as tf

tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)

target_serie = ['a', 'b']
prediction = ['a', 'b']

label_encoder = preprocessing.LabelEncoder()

label_encoder.fit(target_serie)

encoding_prediction = label_encoder.transform(prediction)
display.display(encoding_prediction)


encoding_prediction_one_hot = tf.keras.utils.to_categorical(encoding_prediction, 
                                                            2)
display.display(encoding_prediction_one_hot)
log_loss = metrics.log_loss(target_serie, encoding_prediction_one_hot)
print('Log Loss Metrics: '+ str(log_loss ))
