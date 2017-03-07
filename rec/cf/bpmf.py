import numpy as np
from rec.util.validation import  rmse
from numpy.linalg import inv, cholesky
from scipy.stats import wishart, norm

class Bpmf():
    class FeatureMatrix():
        def __init__(self, num_row, num_col):
            self._num_row = num_row
            self._num_col = num_col
            self.feature_mat = 0.3 * np.random.rand(num_row, num_col)

            self._nu0 = num_col
            self._mu0 = np.zeros(num_col)
            self._beta0 = 2.0
            self._W0 = np.eye(num_col)

            ''' latent variables '''
            self._lambda = np.eye(num_col)
            self._mu = np.zeros(num_col)

        def update_feature(self, f2, alpha, rating_matrix, rating_indices):
            for i, row in enumerate(self.feature_mat):
                f = f2.feature_mat[rating_indices[i], :]
                r_i = rating_matrix[i][rating_indices[i]]
                tmp_lambda = np.linalg.inv(self._lambda + alpha * np.dot(f.T, f))
                mean = (alpha * np.dot(f.T, r_i) + np.dot(self._lambda, self._mu))
                tmp_mu = np.dot(tmp_lambda, mean)
                self.feature_mat[i] = tmp_mu + np.dot(cholesky(tmp_lambda), np.random.randn(self._num_col))

        def update_latent(self, f2):
            N = self._num_row
            s_bar = np.cov(self.feature_mat.T)
            u_bar = np.mean(self.feature_mat, axis=0)

            tmp_mu0 = (self._beta0 * self._mu0 + N * u_bar) / (self._beta0 + N)
            tmp_beta0 = self._beta0 + N
            tmp_nu0 = self._nu0 + N
            tmp_W0 = np.linalg.inv(self._W0) + N * s_bar + (self._beta0 * N /(self._beta0 + N)) * np.dot((self._mu0 - u_bar).T , (self._mu0 - u_bar))
            tmp_W0 = np.linalg.inv(tmp_W0)
            print(tmp_W0)
            #tmp_W0 = (tmp_W0 + tmp_W0.T) / 2.0
            self._lambda = wishart.rvs(tmp_nu0, tmp_W0, 1)
            self._mu = tmp_mu0 + np.dot(cholesky(np.linalg.inv(tmp_beta0 * self._lambda)), np.random.randn(self._num_col))

    def __init__(self, num_user, num_item, num_feature):
        self._user = self.FeatureMatrix(num_user, num_feature)
        self._item = self.FeatureMatrix(num_item, num_feature)
        self._num_user = num_user
        self._num_item = num_item
        self._alpha = 2.0
        self._gibbs = 5
        self._mean_rating = None

    def fit(self, rating_list, batch_size, epoch):
        self._mean_rating = np.mean(rating_list[:,2])
        rating_matrix = np.zeros((self._num_user,self._num_item))
        for rating in rating_list:
            rating_matrix[rating[0]][rating[1]] = rating[2] - self._mean_rating
        predicted = self.predict(rating_list, self._mean_rating)
        print("rmse:{}",rmse(rating_list[:, 2], predicted))
        for _ in range(epoch):
            self._fit(rating_list, rating_matrix)
            predicted = self.predict(rating_list, self._mean_rating)
            print("epoch:{}, rmse:{}", _, rmse(rating_list[:, 2], predicted))
            '''
            batch_index = 0
            while batch_index * batch_size < len(rating_list):
                self._fit(rating_list[rating_index[batch_index * batch_size : (batch_index + 1)* batch_size]])
                batch_index += 1

                predicted = self.predict(rating_list, self._mean_rating)
                print("epoch:{}, rmse:{}", _, rmse(rating_list[:, 2], predicted))
            '''

    def _fit(self, rating_list, rating_matrix):
        user_to_item = [[] for _ in range(self._num_user)]
        item_to_user = [[] for _ in range(self._num_item)]
        for rating in rating_list:
            user_to_item[rating[0]].append(rating[1])
            item_to_user[rating[1]].append(rating[0])
        for _ in range(self._gibbs):
            self._user.update_latent(self._item)
            self._item.update_latent(self._user)
            self._user.update_feature(self._item, self._alpha, rating_matrix, user_to_item)
            self._item.update_feature(self._user, self._alpha, rating_matrix.T, item_to_user)


    def predict(self, rating_list, mean_rating=None):
        if mean_rating == None:
            mean_rating = 0#np.mean(rating_list[:, 2])
        pred = np.sum(self._user.feature_mat[rating_list[:, 0], :] * self._item.feature_mat[rating_list[:, 1], :], axis=1) + mean_rating
        pred[pred > 5] = 5
        pred[pred < 1] = 1
        return pred
