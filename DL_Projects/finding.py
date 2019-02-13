import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

input_size = 784
output_size = 10
hidden_layer_size = 1000

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])

weights_1 = tf.get_variable("weights_1", [input_size, hidden_layer_size])
biases_1 = tf.get_variable("biases_1", [hidden_layer_size])
outputs_1 = tf.nn.relu(tf.matmul(inputs, weights_1) + biases_1)

weights_2 = tf.get_variable("weights_2", [hidden_layer_size, hidden_layer_size])
biases_2 = tf.get_variable("biases_2", [hidden_layer_size])
outputs_2 = tf.nn.relu(tf.matmul(outputs_1, weights_2) + biases_2)

weights_3 = tf.get_variable("weights_3", [hidden_layer_size, hidden_layer_size])
biases_3 = tf.get_variable("biases_3", [hidden_layer_size])
outputs_3  = tf.nn.relu(tf.matmul(outputs_2, weights_3) + biases_3)

weights_4 = tf.get_variable("weights_4", [hidden_layer_size, hidden_layer_size])
biases_4 = tf.get_variable("biases_4", [hidden_layer_size])
outputs_4  = tf.nn.relu(tf.matmul(outputs_3, weights_4) + biases_4)

weights_5 = tf.get_variable("weights_5", [hidden_layer_size, output_size])
biases_5 = tf.get_variable("biases_5", [output_size])
outputs  = tf.matmul(outputs_4, weights_5) + biases_5

loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)
mean_loss = tf.reduce_mean(loss)
optimize = tf.train.AdamOptimizer(learning_rate = 0.0002).minimize(mean_loss)

out_equal_targets = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))
accuracy = tf.reduce_mean(tf.cast(out_equal_targets, tf.float32))

sess = tf.InteractiveSession()
initialize = tf.global_variables_initializer()
sess.run(initialize)

batch_size = 100
batches_number = mnist.train._num_examples // batch_size

max_epoch = 15
pre_validation_loss = 9999999.

for epoch_counter in range(max_epoch):
    curr_epoch_loss = 0
    for batch_counter in range(batches_number):
        input_batch, target_batch = mnist.train.next_batch(batch_size)
        _, batch_loss = sess.run([optimize, mean_loss],
                                 feed_dict={inputs: input_batch, targets: target_batch})
        curr_epoch_loss += batch_loss
    curr_epoch_loss /= batches_number
    input_batch, target_batch = mnist.validation.next_batch(mnist.validation._num_examples)
    validation_loss, validation_accuracy = sess.run([mean_loss, accuracy],
                                                     feed_dict={inputs: input_batch, targets: target_batch})
    print('Epoch '+str(epoch_counter+1)+
          '. Mean loss: '+'{0:.3f}'.format(curr_epoch_loss)+
          '. Validation loss: '+'{0:.3f}'.format(validation_loss)+
          '. Validation accuracy: '+'{0:.2f}'.format(validation_accuracy * 100.)+'%')

    if validation_loss > pre_validation_loss:
        break
    pre_validation_loss = validation_loss
print("End of the training.")

input_batch, target_batch = mnist.test.next_batch(mnist.test._num_examples)
accuracy_test = sess.run([accuracy],
                         feed_dict={inputs: input_batch, targets: target_batch})
accuracy_test_percent = accuracy_test[0] * 100
print('Test accuracy: '+'{0:.2f}'.format(accuracy_test_percent)+'%')