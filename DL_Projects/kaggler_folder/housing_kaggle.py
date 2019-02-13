import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from subprocess import check_output
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)


seed=7

print(check_output(['ls', '/home/wasi/ML_FOLDER/input'])).decode('utf8')

data = pd.read_csv("/home/wasi/ML_FOLDER/input/housingdata.csv", header=None)

print(data.head())

train = data

X = train.iloc[:, 0:13]
Y = train.iloc[:, 13]
# Y=np.reshape(Y, (-1,1))
# scaler = MinMaxScaler()
# print(scaler.fit(X))
# print(scaler.fit(Y))
# xscale=scaler.transform(X)
# yscale=scaler.transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=101)

X_test.to_csv('/home/wasi/ML_FOLDER/input/test_data_set.csv')


def larger_model():
    model = Sequential()
    model.add(Dense(257, input_dim=13, kernel_initializer="glorot_uniform",
                    activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=1000, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X_train, y_train, cv=kfold, n_jobs=1)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [checkpoint]

# pipeline.fit(X_train, y_train)

##Load wights file of the best model :
# wights_file = 'Weights-478--18738.19831.hdf5' # choose the best checkpoint
# pipeline.load_weights(wights_file) # load it
# pipeline.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
#
#
# x_new = np.array([[0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3,
#                    396.9, 4.98]])
#
# predictions = pipeline[1].predict(x_new)
# print("Predicted Value-> {}".format(predictions))



#answer for this is : 24

y_updated = np.array([[0.00632, 18, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.9, 4.98]])

#answer for this is : 34.7

x_updated = np.array([[138, 0.2498, 0, 21.89, 0, 0.624, 5.857, 98.2, 1.6686, 4, 437, 21.2, 392.04]])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
score = pipeline.score(X_test, y_pred)
print('score - >{}'.format(score))
print("Predicted Value-> {}, len->{}".format(len(y_pred), y_pred))
