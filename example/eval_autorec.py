from rec.cf.autorec import AutoRec
from rec.util.data_loader import MovieLens
from rec.util.validation import kfold
import numpy as np
np.set_printoptions(threshold = np.nan)

movie_lens = MovieLens("ml-100k")
for train_index, test_index in kfold(movie_lens.get_sample_num(), 5):
    train = movie_lens.get_matrix(train_index)
    autorec = AutoRec(train.shape[1])
    for _ in range(1000):
        autorec.train(train)
        autorec.print_loss(train, movie_lens.get_list(test_index))

            #print(autorec.encode(train))