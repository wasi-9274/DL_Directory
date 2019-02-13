# # MLP with automatic validation set
# from keras.models import Sequential
# from keras.layers import Dense
# import numpy as np
# # fix random seed for reproducibility
# np.random.seed(7)
# # load pima indians dataset
# dataset = np.loadtxt("/home/wasi/Downloads/unsplash wallpapers/cars.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X=dataset[:,0:5]
# y=dataset[:,5]
# Y=np.reshape(y, (-1,1))
# # create model
# model = Sequential()
# model.add(Dense(12, input_dim=5, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # Fit the model
# model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)



# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("/home/wasi/Downloads/unsplash wallpapers/cars.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:5]
Y = dataset[:,5]
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=5, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
    # evaluate the model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
