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

import os;
path="/home/wasi/Desktop/test"
os.chdir(path)
os.getcwd()

#Variables
dataset=np.loadtxt("/home/wasi/Downloads/unsplash wallpapers/cars.csv", delimiter=",")
print(dataset)
x=dataset[:,0:5]
print(x)
y=dataset[:,5]
y=np.reshape(y, (-1,1))
scaler = MinMaxScaler()
print(scaler.fit(x))
print(scaler.fit(y))
xscale=scaler.transform(x)
yscale=scaler.transform(y)
print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)
model = Sequential()
model.add(Dense(12, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

my_callBacks = [EarlyStopping(monitor='val_acc', patience=5, mode=max)]

# history = model.fit(X_train, y_train, epochs=150, batch_size=30,  verbose=1, callbacks=my_callBacks,
#                     validation_data=(X_test, y_test))

history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)


eval_result = model.evaluate(X_test, y_test)

print("\n\n Test Loss: ", eval_result[0], "Test_accuracy: ", eval_result[1])

# print(history.history.keys())
# # "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
#
# data = pd.read_csv("/home/wasi/ML_FOLDER/ggplot2-master/data-raw/diamonds.csv",
#                    na_values="?", comment="\t", sep=",", skipinitialspace=True)
# dataset = data.copy()
# dataset.tail()
# dataset.isnull().sum()
#
# dataset = dataset.drop(dataset.loc[dataset.x <= 0].index)
# dataset = dataset.drop(dataset.loc[dataset.y <= 0].index)
# dataset = dataset.drop(dataset.loc[dataset.z <= 0].index)
#
# dataset = dataset.dropna()
#
# # x={'Ideal':1,'Premium':2,'Good':3,'Very Good':4, 'Fair': 5}
# # dataset['cut']=dataset['cut'].map(x)
# # dataset
#
# dataset.isnull().sum()
#
# dataset = dataset.drop(['color', 'clarity','cut','depth','table'], axis=1)
#
# print(dataset)
#
# train_dataset = dataset.sample(frac=0.8, random_state=0)
# test_dataset = dataset.drop(train_dataset.index)
#
# print(train_dataset)
#
# train_stats = train_dataset.describe()
# print(train_stats)
# train_stats.pop('price')
# train_stats = train_stats.transpose()
# print(train_stats)
#
# train_labels = train_dataset.pop('price')
# print(train_labels)
# test_labels = test_dataset.pop('price')
#
# # def norm(X):
# #     return(X - train_stats['mean']) / train_stats['std']
# # normed_train_data = norm(train_dataset)
# # normed_test_data = norm(test_dataset)
#
# print(normed_train_data)
#
# def build_model():
#     model = keras.Sequential([
#         layers.Dense(8, activation=tf.nn.relu,
#                      input_shape=[len(train_dataset.keys())]),
#         layers.Dense(5, activation=tf.nn.relu),
#         layers.Dense(1)
#     ])
#     optimizer = tf.keras.optimizers.RMSprop(0.001)
#     model.compile(loss="mse",
#                  optimizer = optimizer,
#                  metrics = ['accuracy'])
#     return model
#
# model = build_model()
#
# model.summary()
#
# example_batch = normed_train_data[: 10]
# example_result = model.predict(example_batch)
# print(example_result)
#
#
# class Dot(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch , logs):
#         if epoch % 100 == 0:
#             print('')
#
#         print('.')
#
# len(train_dataset.keys())
#
# print(type(normed_train_data))
#
# Epochs = 1000
#
# train_label_df = train_labels.to_frame(name='price')
# print(type(train_label_df))
#
# history = model.fit(normed_train_data, train_label_df,
#                    epochs=Epochs, validation_split = 0.2, verbose = 2,
#                    callbacks=[Dot()])
#
# hist = pd.DataFrame(history.history)
#
# hist['epoch'] = history.epoch
# hist.tail()
