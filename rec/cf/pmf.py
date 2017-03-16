"""Probabilistic Matrix Factorization

Reference:
    Probabilistic Matrix Factorization
    R. Salakhutdinov and A.Mnih.
    https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf
    http://www.utstat.toronto.edu/~rsalakhu/code_BPMF/pmf.m
"""

import numpy as np
from rec.util.validation import  rmse

class Pmf():
    def __init__(self, num_user, num_item, num_feature, learning_rate = 1, _lambda = 0.01, momentum = 0.8):
        self._num_user = num_user
        self._num_item = num_item
        self._num_feature = num_feature
        self._user = 0.1 * np.random.rand(num_user, num_feature)
        self._item = 0.1 * np.random.rand(num_item, num_feature)
        self._learning_rate = learning_rate
        self._lambda = _lambda
        self._momentum = momentum

        self._user_inc = np.zeros((num_user, num_feature), dtype = np.float)
        self._item_inc = np.zeros((num_item, num_feature), dtype = np.float)

    def fit(self, rating_list, batch_size, epoch):
        for _ in range(epoch):
            rating_index = np.arange(len(rating_list))
            np.random.shuffle(rating_index)
            batch_index = 0
            while batch_index * batch_size < len(rating_list):
                self._fit(rating_list[rating_index[batch_index * batch_size : (batch_index + 1)* batch_size]])
                batch_index += 1

            predicted = self.predict(rating_list)
            print(rmse(rating_list[:, 2], predicted))

    def _fit(self, rating_list):
        user_index = rating_list[:,0]
        item_index = rating_list[:,1]
        mean_rating = np.mean(rating_list[:, 2]) #average is better?
        error = rating_list[:,2] - (np.sum(self._user[user_index, :] * self._item[item_index,:], axis=1) + mean_rating)

        dr_du = -error[:, np.newaxis] * self._item[item_index,:] + self._lambda * np.abs(self._user[user_index,:])
        dr_di = -error[:, np.newaxis] * self._user[user_index,:] + self._lambda * np.abs(self._item[item_index,:])

        user_grad = np.zeros((self._num_user, self._num_feature))
        item_grad = np.zeros((self._num_item, self._num_feature))

        for i in range(len(rating_list)):
            user_grad[rating_list[i][0]] += dr_du[i,:]
            item_grad[rating_list[i][1]] += dr_di[i,:]

        self._user_inc = self._momentum * self._user_inc + self._learning_rate * user_grad / rating_list.shape[0]
        self._item_inc = self._momentum * self._item_inc + self._learning_rate * item_grad / rating_list.shape[0]

        self._user = self._user - self._user_inc
        self._item = self._item - self._item_inc

    def predict(self, rating_list, mean_rating = None):
        if mean_rating == None:
            mean_rating = np.mean(rating_list[:,2])
        return np.sum(self._user[rating_list[:,0],:] * self._item[rating_list[:,1],:], axis=1) + mean_rating
