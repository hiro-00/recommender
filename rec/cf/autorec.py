import numpy as np
import tensorflow as tf
from rec.util.validation import mask_rmse, list_rmse

class AutoRec():
    def __init__(self, input_size, layer2_size, learning_rate = 0.005, bias = 0.01, regularizers = 0.03):
        self.input_layer = tf.placeholder('float32', [None, input_size], name='input')
        self.unrated_mask = tf.placeholder('float32', [None, input_size], name='unrated_mask')

        w1 = tf.Variable(tf.random_normal(shape=[int(self.input_layer.get_shape()[1]),layer2_size], stddev=0.01))
        b1 = tf.Variable(tf.constant(bias, shape=[layer2_size]))
        self.layer2= tf.nn.relu(tf.add(tf.matmul(self.input_layer, w1), b1))

        w2 = tf.Variable(tf.random_normal(shape=[int(self.layer2.get_shape()[1]), input_size], stddev=0.01))
        b2 = tf.Variable(tf.constant(bias, shape=[input_size]))
        self.layer3 = tf.nn.relu(tf.add(tf.matmul(self.layer2, w2), b2))

        original = tf.mul(self.input_layer, self.unrated_mask)
        self.encoded = tf.mul(self.layer3, self.unrated_mask)
        self.loss = tf.add(tf.reduce_sum(tf.squared_difference(original, self.encoded)), tf.mul(regularizers,
                      tf.add(tf.reduce_sum(tf.mul(original, original)), tf.reduce_sum(tf.mul(self.encoded, self.encoded)))))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss)

        self.session = tf.Session()
        self.session.run(tf.initialize_all_variables())

    def train(self, rating, epoch):
        unrated_mask = np.zeros(rating.shape)
        unrated_mask[rating != -1] = 1
        unrated_mask[rating == -1] = 0
        for _ in range(epoch):
            _, encoded = self.session.run([self.optimizer, self.layer3], feed_dict={self.input_layer: rating,
                                                        self.unrated_mask: unrated_mask})
            encoded[rating == -1] = -1
            print(mask_rmse(rating, encoded))

    def encode(self, rating):
        unrated_mask = np.zeros(rating.shape)
        unrated_mask[rating != -1] = 1
        unrated_mask[rating == -1] = 0
        return self.session.run(self.layer3, feed_dict={self.input_layer: rating,
                                                    self.unrated_mask: unrated_mask})

    def evaluate(self, rating, rating_list):
        print(list_rmse(rating_list, self.encode(rating)))