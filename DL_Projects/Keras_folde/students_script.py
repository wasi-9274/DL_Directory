import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/home/wasi/ML_FOLDER/deep-learning-master/student-admissions/student_data.csv")
print(data[:10])


def plot_points(data):
    X = np.array(data[["gre", "gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected])
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted])
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')


plot_points(data)
plt.show()

data_rank1 = data[data["rank"] == 1]
data_rank2 = data[data["rank"] == 2]
data_rank3 = data[data["rank"] == 3]
data_rank4 = data[data["rank"] == 4]

plot_points(data_rank1)
plt.title("Rank_1")
plt.show()
plot_points(data_rank2)
plt.title("Rank_2")
plt.show()
plot_points(data_rank3)
plt.title("Rank_3")
plt.show()
plot_points(data_rank4)
plt.title("Rank_4")
plt.show()


one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
one_hot_data = one_hot_data.drop('rank', axis=1)
print(one_hot_data[:10])

processed_data = one_hot_data[:]
processed_data['gre'] = processed_data['gre'] / 800
processed_data['gpa'] = processed_data['gpa'] / 4.0
print(processed_data[:10])

sample = np.random.choice(processed_data.index, size=int(len(processed_data) * 0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)
print(len(train_data))
print(len(test_data))
print(train_data[:10])
print(test_data[:10])


features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

print(features[:10])
print(targets[:10])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return  sigmoid(x) * (1-sigmoid(x))


def error_formula(y, output):
    return - y * np.log(output) - (1 - y) * np.log(1 - output)


def error_term_formula(y, output):
    return (y - output) * output * (1 - output)


epochs = 1000
learning_rate = 0.5


def train_nn(features, targets, epochs, learning_rate):
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    weights = np.random.normal(scale=1 / n_features ** 5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            output = sigmoid(np.dot(x, weights))
            error = error_formula(y, output)
            error_term = error_term_formula(y, output)
            del_w += error_term * x

        weights += learning_rate * del_w / n_records
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            print("Epoch : ", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "Warning - loss increaing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=" * 5)
    print("Finished Training!")
    return weights


weights = train_nn(features, targets, epochs, learning_rate)

test_out = sigmoid(np.dot(features_test, weights))
predictions = test_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Predictions accuracy: {:.3f}".format(accuracy))









