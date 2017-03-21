from rec.util.data_loader import MovieLens
from rec.util.validation import rmse
from rec.cf.svd_pp import SvdPp


def test_svd_pp():
    movie_lens = MovieLens("ml-100k")
    model = SvdPp(movie_lens.num_user, movie_lens.num_item, 50)
    model.fit(movie_lens, 9)
    predicted = model.predict(movie_lens)
    assert rmse(movie_lens[:, 2], predicted) < 0.9
