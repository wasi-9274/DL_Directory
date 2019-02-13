import numpy as np
from sklearn import preprocessing

raw_csv_data = np.loadtxt('1.1 Audiobooks_data.csv', delimiter=',')
unscaled_inputs_all = raw_csv_data[:, 1:-1]
targets_all = raw_csv_data[:,-1]

# Balancing the dataset

num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis=0)

# scaling the data
scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)
print(scaled_inputs)

