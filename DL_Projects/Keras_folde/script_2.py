from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVLE'] = "2"


# Helper function

# Plot the data on figure


def plot_data(pl, X, y):
    # plot the class where y == 0
    pl.plot(X[y == 0, 0], X[y == 0, 1], 'ob', alpha=0.5)
    # plot the class where y == 1
    pl.plot(X[y == 1, 0], X[y == 1, 1], 'xr', alpha=0.5)
    pl.legend(['0', '1'])
    return pl


# Common function that draws the decision boundaries

def plot_decision_boundary(model, X, y):

    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)

    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]

    # Make predictions with the model and reshape the output so contour can plot it

    c = model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12, 8))

    # plot the contour
    plt.contourf(aa, bb, Z, cmap='bwr', alpha=0.2)

    # plot the moons of the data

    plot_data(plt, X, y)

    return plt


X, y = make_circles(n_samples=1000, shuffle=True, noise=0.1, random_state=None, factor=0.6)
# X, y = make_blobs(n_samples=1000, centers=2, random_state=42)

# pl1 = plot_data(plt, X1, y1)
# pl1.show()
pl = plot_data(plt, X, y)
pl.show()


# Split the data into training and test setsX, y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating the keras model

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(5, input_shape=(2, ), activation="tanh", name="Hidden_1"))
model.add(Dense(4, activation="tanh", name="Hidden_2"))
model.add(Dense(4, activation="tanh", name="Hidden_3"))
model.add(Dense(1, activation="sigmoid", name="Output_layer"))
model.summary()
model.compile(Adam(lr=0.05), loss="binary_crossentropy", metrics=['accuracy'])

from keras.callbacks import EarlyStopping
my_callBacks = [EarlyStopping(monitor='val_acc', patience=5, mode=max)]

model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callBacks, validation_data=(X_test, y_test))
# from keras.utils import plot_model
# plot_model(model, to_file="model.png", show_layer_names=True, show_shapes=True)
eval_result = model.evaluate(X_test, y_test)

print("\n\n Test Loss: ", eval_result[0], "Test_accuracy: ", eval_result[1])
plot_decision_boundary(model, X, y).show()