import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from keras import models
from keras.optimizers import Adam, SGD
from sklearn.linear_model import LinearRegression
import seaborn as sns

import os;
path="/home/wasi/Desktop/test"
os.chdir(path)
os.getcwd()

#Variables
# /home/wasi/Desktop/diamonds_req.csv
dataset=np.loadtxt("/home/wasi/Desktop/diamonds_req.csv", delimiter=",")
x=dataset[:, 0:4]
y=dataset[:, 4]
y=np.reshape(y, (-1, 1))
scaler = MinMaxScaler()
print(scaler.fit(x))
print(scaler.fit(y))
xscale=scaler.transform(x)
yscale=scaler.transform(y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale, test_size=.3, random_state=101)

# Sequential model with keras

model = Sequential()
model.add(Dense(1, input_dim=4, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

# model.load_weights("/home/wasi/Desktop/test/car_model.h5")

my_callBacks = [EarlyStopping(monitor='val_loss', patience=10, mode=max)]

history = model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callBacks, validation_data=(X_test, y_test))

model.save('car_model.h5')

print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333
# 22854
# Xnew = np.array([[57, 0, 34, 58481,7054]])
# Xnew = np.array([[39, 1, 28, 0,3282]])


Xnew = np.array([[0.23, 3.95, 3.98, 2.43]])
ynew=model.predict(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
