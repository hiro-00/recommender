import scipy.sparse as sparse
import numpy as np
import os


class MovieLens():
    def __init__(self, dataset, row_num = None):
        path = os.path.dirname(__file__) + "/../../data/" + dataset + "/ratings.dat"
        self._data = np.loadtxt(path, delimiter="::", dtype=int)
        self._num_user = len(np.unique(self._data[:,0])) + 1
        self._num_movie = np.max(self._data[:,1]) + 1
        if row_num is not None:
            self._data = self._data[:row_num]

    def get_matrix(self, indices):
        rating = np.full((self._num_user, self._num_movie), -1)
        for d in self._data[indices]:
            rating[d[0]][d[1]] = d[2]
        return rating

    def get_sparse(self, indices = None):
        return sparse.csr_matrix((self._data[:,2],
                                  (self._data[:,0], self._data[:,1])),
                                 shape=(self._num_user, self._num_movie))

    def get_list(self, indices = None):
        if indices == None:
            return self._data[:, 0:3]
        return self._data[indices, 0:3]


    @property
    def num_user(self):
        return self._num_user

    @property
    def num_item(self):
        return self._num_movie

    def __len__(self):
        return len(self._data)

    def __getitem__(self, position):
        return self._data[position]

test = MovieLens("ml-100k")
#print(test._data)
train_fold=[0,1,2,3]
test_fold=[4,5,6]
train_array = test.get_matrix(train_fold)
print(test[test_fold])
print(test[test_fold][:,1])
#rating = test.rating_to_np()
#rating[rating != -1]=1
#rating[rating == -1]=0

#print(rating)
