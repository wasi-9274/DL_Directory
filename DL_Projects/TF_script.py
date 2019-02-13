# import numpy as np
#
# observations = 1000
# xs = np.random.uniform(-10, 10, size=(observations, 1))
# zs = np.random.uniform(-10, 10, size=(observations, 1))
# inputs = np.column_stack((xs, zs))
#
# noise = np.random.uniform(-1, 1, size=(observations, 1))
# targets = 2 * xs + 3 * zs + 5 + noise
#
# weights = np.random.uniform(-0.1, 0.1, size=(2, 1))
# biases = np.random.uniform(-0.1, 0.1, size=1)
#
# learning_rate = 0.02
#
# for i in range(1000):
#     outputs = np.dot(inputs, weights) + biases
#     deltas = outputs - targets
#     loss = np.sum(deltas ** 2) / 2 / observations
#     print(loss)
#     deltas_scaled = deltas / observations
#     weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
#     biases = biases - learning_rate * np.sum(deltas_scaled)
#
# print(weights)
# print(biases)

import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


observations = 1000
X = np.random.uniform(-10, 10, size=(observations, 1))
y = np.random.uniform(-10, 10, size=(observations, 1))
# inputs = np.column_stack((X, y))
# print(inputs.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# noise = np.random.uniform(-1, 1, size=(observations, 1))
# targets = 2 * x + 3 * y + 5 + noise

weights = np.random.uniform(-0.1, 0.1, size=(2, 1))
biases = np.random.uniform(-0.1, 0.1, size=1)

model = Sequential()
model.add(Dense(5, input_shape=(1, ), activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="relu"))
model.summary()
model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=1)
eval_result = model.evaluate(X_test, y_test)
print("\n\n Test Loss: ", eval_result[0], "Test_accuracy: ", eval_result[1])




