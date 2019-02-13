import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

observations = 1000
xs = np.random.uniform(-10, 10, size=(observations, 1))
zs = np.random.uniform(-10, 10, size=(observations, 1))
inputs = np.column_stack((xs, zs))

noise = np.random.uniform(-1, 1, size=(observations, 1))
targets = 2 * xs + 3 * zs + 4 + noise

weights = np.random.uniform(-0.1, 0.1, size=(2, 1))
biases = np.random.uniform(-0.1, 0.1 , size=1)

learning_rate = 0.02

for i in range(500):
    outputs = np.dot(inputs, weights) + biases
    deltas = outputs - targets
    loss = np.sum(deltas ** 2) / 2 / observations
    print(loss)
    deltas_scaled = deltas / observations
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)

print(weights)
print(biases)