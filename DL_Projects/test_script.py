import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

input_size = 784
output_size = 10
hidden_layer_size = 50

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])

weights_1 = tf.get_variable('weights_1', [input_size, hidden_layer_size])
biases_1 = tf.get_variable('biases_1', [hidden_layer_size])
outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable('weights_2', [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable('biases_2', [hidden_layer_size])
outputs_2 = tf.nn.relu(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable('weights_3', [hidden_layer_size, output_size])
biases_3 = tf.get_variable('biases_3', [output_size])
outputs = tf.matmul(outputs_2, weights_3) + biases_3

loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
mean_loss = tf.reduce_mean(loss)

optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mean_loss)

out_equals_target = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(out_equals_target, tf.float32))

batch_size = 100

batches_number = mnist.train._num_examples // batch_size

max_epoch = 15

pre_validation_loss = 9999999.

sess = tf.InteractiveSession()

initialize = tf.global_variables_initializer()

sess.run(initialize)

for epoch_counter in range(max_epoch):
    curr_epoch_loss = 0
    for batch_counter in range(batches_number):
        inputs_batch, target_batch = mnist.train.next_batch(batch_size)
        _, batch_loss = sess.run([optimize, mean_loss],
                                 feed_dict={inputs: inputs_batch, targets: target_batch})
        curr_epoch_loss += batch_loss

    curr_epoch_loss /= batches_number
    inputs_batch, target_batch = mnist.validation.next_batch(mnist.validation._num_examples)
    validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
                                                    feed_dict={inputs: inputs_batch, targets: target_batch})
    print('Epoch ' + str(epoch_counter+1) +
          '. Training loss: '+'{0:.3f}'.format(curr_epoch_loss) +
          '. Validation_loss: '+'{0:.3f}'.format(validation_loss) +
          '. Validation_accuracy: '+'{0:.3f}'.format(validation_accuracy * 100) + '%')

    if validation_loss > pre_validation_loss:
        break
    pre_validation_loss = validation_loss

print("End of training")

inputs_batch, target_batch = mnist.test.next_batch(mnist.test._num_examples)
test_accuracy = sess.run([accuracy],
                         feed_dict={inputs: inputs_batch, targets: target_batch})
test_accuracy_percentage = test_accuracy[0] * 100
print('Test Accuracy is: '+'{0:.2f}'.format(test_accuracy_percentage) + '%')
