# importing the relevant libraries

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate the random input data to train on
observations = 1000

xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(-10, 10, size=(observations, 1))

inputs = np.column_stack([xs, zs])

print(inputs.shape)

# creating the targets we will aim at

noise = np.random.uniform(-1, 1, (observations, 1))
targets = 2 * xs + 3 * zs + 5 + noise
print(targets.shape)

# Plotting the training data

targets = targets.reshape(observations, )
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, zs, targets)
ax.set_xlabel('xs')
ax.set_ylabel('zs')
ax.set_zlabel('Targets')
ax.view_init(azim=100)
plt.show()
targets = targets.reshape(observations, 1)

# initializing variables:

init_range = 0.1

weights = np.random.uniform(-init_range, init_range, size=(2, 1))
biases = np.random.uniform(-init_range, init_range, size=1)

print(weights)
print(biases)

# setting a learning rate

learning_rate = 0.02

# Training our model

for i in range(1030):
    outputs = np.dot(inputs, weights) + biases
    deltas = outputs - targets
    loss = np.sum(deltas **2) /2 /observations
    print(loss)
    deltas_selected = deltas / observations
    weights = weights - learning_rate * np.dot(inputs.T, deltas_selected)
    biases = biases - learning_rate * np.sum(deltas_selected)

# Print weights and biases to see if it correctly works

print(weights)
print(biases)

# Plot last output vs targets

plt.plot(outputs, targets)
plt.xlabel('Outputs')
plt.ylabel('Targets')
plt.show()


