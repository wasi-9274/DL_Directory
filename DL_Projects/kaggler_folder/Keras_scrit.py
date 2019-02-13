import tensorflow as tf
import numpy as np
from keras import backend as kbe



print("tensorflow version: " + str(tf.__version__))
hello = tf.constant("hello from tensorflow")
sess = tf.Session()
print(sess.run(hello))


data = kbe.variable(np.random.random((4, 2)))
zero_data = kbe.ones_like(data)
print(kbe.eval(zero_data))