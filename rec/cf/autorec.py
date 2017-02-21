import numpy as np
import tensorflow as tf

class AutoRec():
    def __init__(self, input_size):
        layer1_size = 5
        learning_rate = 0.1

        self.input_layer = tf.placeholder('float32', [None, input_size], name='input')
        self.unrated_mask = tf.placeholder('float32', [None, input_size], name='unrated_mask')

        w1 = tf.Variable(tf.random_normal(shape=[int(self.input_layer.get_shape()[1]),layer1_size], stddev=0.01))
        b1 = tf.Variable(tf.constant(0.1, shape=[layer1_size]))
        layer1 = tf.nn.relu(tf.add(tf.matmul(self.input_layer, w1), b1))

        w2 = tf.Variable(tf.random_normal(shape=[int(layer1.get_shape()[1]), input_size], stddev=0.01))
        b2 = tf.Variable(tf.constant(0.1, shape=[input_size]))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, w2), b2))

        original = tf.mul(self.input_layer, self.unrated_mask)
        encoded = tf.mul(layer2, self.unrated_mask)
        loss = tf.add(tf.squared_difference(original, encoded), tf.mul(learning_rate,
                      tf.add(tf.reduce_sum(tf.mul(original, original)), tf.reduce_sum(tf.mul(encoded, encoded)))))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train(self, rating):
        unrated_mask = np.zeros(rating.shape)
        unrated_mask[rating != -1] = 1
        unrated_mask[rating == -1] = 0
        self.session.run(self.optimizer, feed_dict={self.input_layer: rating,
                                                    self.unrated_mask: unrated_mask})
