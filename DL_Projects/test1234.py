import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

observations = 1000

xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(low=-10, high=10, size=(observations, 1))

inputs = np.stack([xs, zs])
print(inputs)

noise = np.random.uniform(-1, 1, (observations, 1))
targets = 2 * xs + 3 * zs + 5 + noise
print(targets.shape)

targets = targets.reshape(observations, )
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("xs")
ax.set_ylabel('zs')
ax.set_zlabel("targets")
ax.view_init(azim=100)
plt.show()
targets = targets.reshape(observations, 1)

init_range = 0.1

weights = np.random.uniform(-init_range, init_range, (2, 1))
biases = np.random.uniform(-init_range, init_range, 1)

print(weights)
print(biases)

learning_rate = 0.02

for i in range(100):
    outputs = np.dot(inputs, weights) + biases
    deltas = outputs - targets
    loss = np.sum(deltas ** 2) / 2 / observations
    print(loss)
    deltas_scaled = deltas / observations
    weights = weights - learning_rate * np.dot(inputs.T, deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)


print(weights)
print(biases)

plt.plot(outputs, targets)
plt.xlabel('Outputs')
plt.ylabel('Targets')
plt.show()
