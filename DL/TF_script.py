import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Data generation
observations = 10000

xs = np.random.uniform(low = -10, high= 10, size=(observations, 1))
zs = np.random.uniform(-10, 10, (observations, 1))

generated_inputs = np.column_stack((xs, zs))

noise = np.random.uniform(-1, 1, (observations, 1))

generated_targets = 2 * xs - 3 * zs + 5 + noise

np.savez('Tf_intro', inputs=generated_inputs, targets = generated_targets)

# solving with the tensorflow

input_size = 2
output_size = 1

# Outlining the model

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])

weights = tf.Variable(tf.random_uniform([input_size, output_size], minval=-0.1, maxval=0.1))
biases = tf.Variable(tf.random_uniform([output_size], minval=-0.1, maxval=0.1))
outputs = tf.matmul(inputs, weights) + biases

# Defining objective functions

mean_loss = tf.losses.mean_squared_error(labels=targets, predictions=outputs) / 2.

optimize = tf.train.AdamOptimizer(learning_rate=0.09).minimize(mean_loss)

# Preparing for execution

sess = tf.InteractiveSession()

# Initializing the variables
initializer = tf.global_variables_initializer()
sess.run(initializer)

# Loading the training_data

training_data = np.load('Tf_intro.npz')

for e in range(100):
    _, curr_loss = sess.run([optimize, mean_loss], feed_dict={inputs: training_data['inputs'],
                                                              targets: training_data['targets']})
    print(curr_loss)


out = sess.run([outputs],
               feed_dict={inputs: training_data['inputs']})

plt.plot(np.squeeze(out), np.squeeze(training_data['targets']))
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()






