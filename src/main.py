import numpy as np
import pandas as pd

import tensorflow as tf, tensorflow_datasets as tfds

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

emnist = tfds.load('emnist', data_dir='emnist')
(x_train, y_train), (x_test, y_test) = emnist['train'], emnist['test']

x_train, x_test = x_train / 255.0, x_test / 255.0
 