import numpy as np
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

batch_size = 128
df = pd.read_csv("train.csv")
y = df.label.values
X = df.drop("label", axis=1).values
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

model = Sequential()
model.add(Convolution2D(32, (6, 6), activation="relu", input_shape=(28, 28, 1)))
model.add(Convolution2D(64, (6, 6), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, epochs=5, verbose=1, validation_data=(X_test, Y_test))
model.save('model_new.h5')
score = model.evaluate(X_test, Y_test, verbose=0)
print("Test loss: {} ".format(score[0]))
print("Test accuracy: {}".format(score[1]))

test_data = pd.read_csv("test.csv")
import matplotlib.pyplot as plt
ans = []
for i in range(len(test_data.as_matrix())):
    img = test_data.as_matrix()[i]
    img = img / 255
    img = np.array(img).reshape((28, 28, 1))
    img = np.expand_dims(img, axis=0)
    img_class = model.predict_classes(img)
    ans.append(img_class)
ids = [i+1 for i in range(len(ans))]
df_1 = pd.DataFrame({"ImageId" : ids, "Label" : ans})
df_1.to_csv("1.csv", index = False)


#print(classes[0:10])
print(img_class)
#prediction = img_class[0]
classname = img_class[0]
print("Predicted number is: ",classname)

img = img.reshape((28,28))
plt.imshow(img)
plt.title(classname)
plt.show()



