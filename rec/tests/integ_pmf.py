from rec.util.data_loader import MovieLens
from rec.util.validation import rmse
from rec.cf.pmf import Pmf


def test_pmf():
    movie_lens = MovieLens("ml-100k", 10000)
    pmf = Pmf(movie_lens.num_user, movie_lens.num_item, 10)
    pmf.fit(movie_lens, int(len(movie_lens) / 10), 70)
    predicted = pmf.predict(movie_lens)
    assert rmse(movie_lens[:, 2], predicted) < 0.87