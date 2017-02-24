import pandas as pd
import numpy as np
import os

class MovieLens():
    def __init__(self, dataset):
        path = os.path.dirname(__file__) + "/../../data/" + dataset + "/ratings.dat"
        self._data = np.loadtxt(path, delimiter="::", dtype=int)
        self._num_user = len(np.unique(self._data[:,0]))
        self._num_movie = np.max(self._data[:,1])#len(np.unique(self._data[:,1]))

    def get_matrix(self, indices):
        result = np.full((self._num_user, self._num_movie), -1)
        for d in self._data[indices]:
            result[d[0]-1][d[1]-1] = d[2]
        return result

    def get_list(self, indices):
        return self._data[indices, 0:3]

    def get_sample_num(self):
        return len(self._data)


    def get_sparse_array(self):
        pass


test = MovieLens("ml-100k")
train_fold=[0,1,2,3]
test_fold=[4,5,6]
train_array = test.get_matrix(train_fold)
print(test.get_list(test_fold))
#rating = test.rating_to_np()
#rating[rating != -1]=1
#rating[rating == -1]=0

#print(rating)
