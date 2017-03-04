
from rec.util.data_loader import MovieLens
from rec.util.validation import kfold, rmse
import numpy as np
np.set_printoptions(threshold = np.nan)

def eval_autorec():
    from rec.cf.autorec import AutoRec
    movie_lens = MovieLens("ml-100k")
    for train_index, test_index in kfold(movie_lens.get_sample_num(), 5):
        train = movie_lens.get_matrix(train_index)
        autorec = AutoRec(train.shape[1], train.shape[1]//5)
        autorec.train(train, 3)

def eval_pmf():
    from rec.cf.pmf import Pmf
    movie_lens = MovieLens("ml-100k")
    for train_index, test_index in kfold(movie_lens.get_sample_num(), 5):
        pmf = Pmf(movie_lens.get_num_user(), movie_lens.get_num_item(), 5)
        train = movie_lens.get_list(train_index)
        pmf.fit(train, train.shape[0]/10, 300)
        test = movie_lens.get_list(test_index)
        predicted = pmf.predict(test)
        print("--")
        print(rmse(test[:,2], predicted))
        break
eval_pmf()