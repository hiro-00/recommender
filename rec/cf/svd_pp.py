"""SVD++

Reference:
    Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model
    Yehuda Koren
    http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf
"""

import numpy as np
from rec.util.validation import  rmse


class SvdPp:
    def __init__(self, num_user, num_item, num_feature):
        self._num_user = num_user
        self._num_item = num_item
        self._num_feature = num_feature
        self._mu = 0
        self._b_u = np.zeros(num_user)
        self._b_i = np.zeros(num_item)
        self._num_r_u = np.zeros(num_user)
        self._c = 0.1 * np.random.rand(num_user, num_item)

        self._p_u = 0.01 * np.random.rand(num_user, num_feature)
        self._q_i = 0.01 * np.random.rand(num_item, num_feature)

        self._lambda_b = 0.005
        self._gamma_b = 0.007
        self._lambda_pq = 0.015
        self._gamma_pq = 0.007
        self._lambda_wc = 0.015
        self._gamma_wc = 0.001

        self._N_u = [[] for _ in range(self._num_user)]
       # self._R_i_u =
        self._y_i = 0.1 * np.random.rand(self._num_item, self._num_feature)

    def fit(self, rating_list, epoch):
        self._mu = np.mean(rating_list[:,2])
        for rating in rating_list:
            self._N_u[rating[0]].append(rating[1])
        for iter in range(epoch):
            for rating in rating_list:
                user_index = rating[0]
                item_index = rating[1]
                rating_value = rating[2]
                predicted = self.predict(rating.reshape(1,rating.shape[0]))
                error = np.sum(rating_value - predicted)
                self._b_u[user_index] += self._gamma_b * (error - self._lambda_b * self._b_u[user_index])
                self._b_i[item_index] += self._gamma_b * (error - self._lambda_b * self._b_i[item_index])
                self._p_u[user_index] += self._gamma_pq * (error * self._q_i[item_index] - self._lambda_pq * self._p_u[user_index])
                n_u = self._N_u[user_index]
                self._q_i[item_index] += self._gamma_pq * (error *
                            (self._p_u[user_index] + np.sum(self._y_i[n_u], axis=0)/np.sqrt(len(n_u)))
                                                            - self._lambda_pq * self._q_i[item_index])
                self._y_i[n_u] += self._gamma_pq * (error * self._q_i[item_index]/np.sqrt(len(n_u)) - self._lambda_pq * self._y_i[n_u])
            print("epoch:{}, rmse:{}", iter, rmse(rating_list[:, 2], self.predict(rating_list, True)))

    def predict(self, rating_list, eval=False):
        user_indices = rating_list[:,0]
        item_indices = rating_list[:,1]
        baseline = self._mu + self._b_u[user_indices] + self._b_i[item_indices]
        pred = np.zeros(len(rating_list))
        for i in range(len(rating_list)):
            user_index = rating_list[i,0]
            item_index = rating_list[i,1]
            n_u = self._N_u[user_index]
            latent = np.dot(self._q_i[item_index],
                               (self._p_u[user_index] + np.sum(
                                   self._y_i[n_u], axis=0)/np.sqrt(len(n_u))).T)
            pred[i] = baseline[i] + latent
        pred[pred > 5] = 5
        pred[pred < 1] = 1
        return pred
