import numpy as np
import tensorflow as tf


class Audiobooks_Data_Reader():

    def __init__(self, dataset, batch_size = None):
        npz = np.load('Audiobooks_data_{0}.npz'.format(dataset))
        self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
        if batch_size is None:
            self.batch_size = self.inputs.shape[0]
        else:
            self.batch_size = batch_size
        self.curr_batch = 0
        self.batch_count = self.inputs.shape[0] // self.batch_size

    def __next__(self):
        if self.curr_batch >= self.batch_count:
            self.curr_batch = 0
            raise StopIteration()
        batch_slice = slice(self.curr_batch * self.batch_size, (self.curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self.curr_batch +=1

        classes_num = 2
        targets_one_hot = np.zeros((targets_batch.shape[0], classes_num))
        targets_one_hot[range(targets_batch.shape[0]), targets_batch] = 1
        return inputs_batch, targets_one_hot

    def __iter__(self):
        return  self

input_size = 10
output_size = 2
hidden_layer_size = 50

inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.int32, [None, output_size])

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

max_epoch = 50

pre_validation_loss = 9999999.

sess = tf.InteractiveSession()

initialize = tf.global_variables_initializer()

sess.run(initialize)

train_data = Audiobooks_Data_Reader('train', batch_size)
validation_data = Audiobooks_Data_Reader('validation')

for epoch_counter in range(max_epoch):
    curr_epoch_loss = 0.
    for input_batch, target_batch in train_data:
        _, batch_loss = sess.run([optimize, mean_loss],
                                 feed_dict={inputs: input_batch, targets: target_batch})
        curr_epoch_loss += batch_loss
    curr_epoch_loss /= train_data.batch_count
    validation_loss = 0.
    validation_accuracy = 0.
    for input_batch, target_batch in validation_data:
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


test_data = Audiobooks_Data_Reader('test')
for input_batch, target_batch in test_data:
    test_accuracy = sess.run([accuracy], feed_dict={inputs: input_batch, targets: target_batch})
    test_accuracy_percent = test_accuracy[0] * 100
    print('Test Accuracy is: '+'{0:.2f}'.format(test_accuracy_percent) + '%')


