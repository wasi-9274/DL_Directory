import numpy as np
import tensorflow as tf
from sklearn import preprocessing

# Extract the dataset from the csv
raw_csv_data = np.loadtxt('Audiobooks_data.csv', delimiter=",")
unscaled_inputs_all = raw_csv_data[:, 1:-1]
targets_all = raw_csv_data[:, -1]

# Balance the dataset
num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter +=1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)
unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_prior = np.delete(targets_all, indices_to_remove, axis=0)

# Standardize the inputs
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

# Shuffle the data
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_prior[shuffled_indices]

# Split the dataset into , train, validation, test
samples_count = shuffled_inputs.shape[0]

train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count + validation_samples_count:]
test_targets = shuffled_targets[train_samples_count + validation_samples_count:]

print(np.sum(train_targets), train_samples_count, np.sum(train_targets)/train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets)/ validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets)/test_samples_count)


# Save the three dataset

np.savez('Audiobooks_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Audiobooks_data_validation', inputs=validation_inputs,targets=validation_targets)
np.savez('Audiobooks_data_test', inputs=test_inputs, targets=test_targets)


