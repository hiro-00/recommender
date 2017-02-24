from rec.cf.autorec import AutoRec
from rec.util.data_loader import MovieLens
from rec.util.validation import kfold
import numpy as np
np.set_printoptions(threshold = np.nan)

movie_lens = MovieLens("ml-100k")
for train_index, test_index in kfold(movie_lens.get_sample_num(), 5):
    train = movie_lens.get_matrix(train_index)
    autorec = AutoRec(train.shape[1], train.shape[1]//5)
    autorec.train(train, 3)