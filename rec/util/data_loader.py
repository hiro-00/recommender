import pandas as pd
import numpy as np
import os

class MovieLens():
    def __init__(self, dataset):
        path = os.path.dirname(__file__) + "/../../data/" + dataset + "/ratings.dat"
        self._data = np.loadtxt(path, delimiter="::", dtype=int)
        self._num_user = len(np.unique(self._data[:,0]))
        self._num_movie = len(np.unique(self._data[:,1]))

    def get_np_array(self, indices):
        result = np.full((self._num_user, self._num_movie), -1)
        for d in self._data[indices]:
            result[d[0]][d[1]] = d[2]
        return result

    def get_raw_data(self, indices):
        return self._data[indices]


    def get_sparse_array(self):
        pass


test = MovieLens("ml-100k")
train_fold=[0,1,2,3]
test_fold=[4,5,6]
train_array = test.get_np_array(train_fold)
print(train_array)
test_array=test.get_raw_data(test_fold)
print(test_array)

#rating = test.rating_to_np()
#rating[rating != -1]=1
#rating[rating == -1]=0

#print(rating)
