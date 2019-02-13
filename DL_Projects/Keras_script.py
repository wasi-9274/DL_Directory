import tensorflow as tf


# print("tensorflow version: " + str(tf.__version__))
hello = tf.constant("hello from tensorflow")
sess = tf.Session()
print(sess.run(hello))
