import tensorflow as tf
from PIL import Image
import keras
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.preprocessing import image

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)
print(train_x.shape)
print(test_x.shape)

train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784, )))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation="softmax"))
model.summary()
model.compile(optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
model.load_weights("/home/wasi/Desktop/scrapy_projects/DL_Projects/Project/mnistmodel.h5")
#
# my_callBacks = [EarlyStopping(monitor='val_acc', patience=5, mode=max)]
# model.fit(train_x, train_y, batch_size=32, epochs=15, verbose=1,
#           callbacks=my_callBacks, validation_data=(test_x, test_y))
# model.save("mnistmodel.h5")
# accuracy = model.evaluate(x=test_x, y=test_y, batch_size=32)
# print("Accuracy of the model is --> {}".format(accuracy[1] * 100))

img = image.load_img(path="/home/wasi/Downloads/1_0D7K4JZNABjK2RMQyM5zVQ.png", color_mode='grayscale',
                     target_size=(28, 28, 1))
img = image.img_to_array(img)
test_img = img.reshape((1, 784))
img_class = model.predict_classes(test_img)
prediction = img_class[0]
classname = img_class[0]
print("Class: ", classname)
img = img.reshape((28, 28))
plt.imshow(img)
plt.title(prediction)
plt.show()
