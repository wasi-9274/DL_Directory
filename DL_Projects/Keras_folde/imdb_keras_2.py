import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

np.random.seed(42)


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)
print(x_train.shape)
print(x_test.shape)


print(x_train[0])
print(y_train[0])

tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(x_train[0])