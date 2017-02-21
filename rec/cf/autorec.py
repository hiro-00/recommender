import numpy as np
import tensorflow as tf

input_size = 20
layer1_size = 5
learning_rate = 0.1

input_layer = tf.placeholder('float32', [None, input_size], name='input')

w1 = tf.Variable(tf.random_normal(shape=[int(input_layer.get_shape()[1]),layer1_size], stddev=0.01))
b1 = tf.Variable(tf.constant(0.1, shape=[layer1_size]))
layer1 = tf.nn.relu(tf.add(tf.matmul(input_layer, w1), b1))


w2 = tf.Variable(tf.random_normal(shape=[int(layer1.get_shape()[1]), input_size], stddev=0.01))
b2 = tf.Variable(tf.constant(0.1, shape=[input_size]))
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2))

loss = tf.add(tf.squared_difference(input_layer, layer2), tf.mul(learning_rate,
              tf.add(tf.reduce_sum(tf.mul(input_layer,input_layer)), tf.reduce_sum(tf.mul(layer2,layer2)))))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)


input_array = np.zeros((10, input_size))

session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(optimizer, feed_dict={input_layer: input_array})
