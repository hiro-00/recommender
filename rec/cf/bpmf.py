"""Bayesian Probabilistic Matrix Factorization

Reference:
    Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo
    R. Salakhutdinov and A.Mnih.
    http://www.cs.toronto.edu/~rsalakhu/papers/bpmf.pdf
    http://www.utstat.toronto.edu/~rsalakhu/code_BPMF/bayespmf.m
"""

import numpy as np
import scipy.sparse as sparse
from rec.util.validation import  rmse
from numpy.linalg import inv, cholesky
from scipy.stats import wishart, norm


class Bpmf():
    class FeatureMatrix():
        def __init__(self, num_row, num_col, beta = 2.0):
            self._num_row = num_row
            self._num_col = num_col
            self._feature_matrix = 0.3 * np.random.rand(num_row, num_col)
            self._nu0 = num_col
            self._mu0 = np.zeros(num_col)
            self._beta0 = beta
            self._W0 = np.eye(num_col)

            ''' latent variables '''
            self._lambda = np.eye(num_col)
            self._mu = np.zeros(num_col)

        @property
        def feature_matrix(self):
            return self._feature_matrix

        def update_feature(self, f2, alpha, rating_matrix):
            for i, row in enumerate(self.feature_matrix):
                indices = rating_matrix[i, :].indices
                f = f2.feature_matrix[indices, :]
                r_i = rating_matrix[i, :].data
                tmp_lambda = np.linalg.inv(self._lambda + alpha * np.dot(f.T, f))
                tmp_mean = (alpha * np.dot(f.T, r_i) + np.dot(self._lambda, self._mu))
                tmp_mu = np.dot(tmp_lambda, tmp_mean)
                self.feature_matrix[i] = tmp_mu + np.dot(cholesky(tmp_lambda), np.random.randn(self._num_col))

        def update_latent(self, f2):
            N = self._num_row
            s_bar = np.cov(self.feature_matrix.T)
            u_bar = np.mean(self.feature_matrix, axis=0)

            tmp_mu0 = (self._beta0 * self._mu0 + N * u_bar) / (self._beta0 + N)
            tmp_beta0 = self._beta0 + N
            tmp_nu0 = self._nu0 + N
            tmp_W0 = np.linalg.inv(self._W0) + N * s_bar + (self._beta0 * N /(self._beta0 + N)) * np.dot((self._mu0 - u_bar).T , (self._mu0 - u_bar))
            tmp_W0 = np.linalg.inv(tmp_W0)
            #tmp_W0 = (tmp_W0 + tmp_W0.T) / 2.0

            self._lambda = wishart.rvs(tmp_nu0, tmp_W0, 1)
            self._mu = tmp_mu0 + np.dot(cholesky(np.linalg.inv(tmp_beta0 * self._lambda)), np.random.randn(self._num_col))

    def __init__(self, num_user, num_item, num_feature, alpha = 2.0, gibbs_iter = 5):
        self._user = self.FeatureMatrix(num_user, num_feature, alpha)
        self._item = self.FeatureMatrix(num_item, num_feature, alpha)
        self._num_user = num_user
        self._num_item = num_item
        self._alpha = alpha
        self._gibbs_iter = gibbs_iter
        self._mean_rating = None

    def _fit(self, csr, csc):
        for _ in range(self._gibbs_iter):
            self._user.update_latent(self._item)
            self._item.update_latent(self._user)
            self._user.update_feature(self._item, self._alpha, csr)
            self._item.update_feature(self._user, self._alpha, csc.T)

    def fit(self, rating_list, epoch):
        self._mean_rating = np.mean(rating_list[:,2])
        csr = sparse.csr_matrix((rating_list[:,2] - self._mean_rating,
                                  (rating_list[:,0], rating_list[:,1])),
                                 shape=(self._num_user, self._num_item))
        csc = csr.tocsc()
        predicted = self.predict(rating_list, self._mean_rating)
        print("rmse:{}",rmse(rating_list[:, 2], predicted))
        for _ in range(epoch):
            self._fit(csr, csc)
            predicted = self.predict(rating_list, self._mean_rating)
            print("epoch:{}, rmse:{}", _, rmse(rating_list[:, 2], predicted))

    def predict(self, rating_list, mean_rating = None):
        if mean_rating is None:
            mean_rating = np.mean(rating_list[:, 2])
        pred = np.sum(self._user.feature_matrix[rating_list[:, 0], :] * self._item.feature_matrix[rating_list[:, 1], :], axis=1) + mean_rating
        pred[pred > 5] = 5
        pred[pred < 1] = 1
        return pred
