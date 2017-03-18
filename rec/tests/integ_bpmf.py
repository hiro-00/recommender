from rec.util.data_loader import MovieLens
from rec.util.validation import rmse
from rec.cf.bpmf import Bpmf


def test_bmpf():
    movie_lens = MovieLens("ml-100k", 10000)
    bpmf = Bpmf(movie_lens.num_user, movie_lens.num_item, 10)
    bpmf.fit(movie_lens, 6)
    predicted = bpmf.predict(movie_lens)
    assert rmse(movie_lens[:, 2], predicted) < 0.87