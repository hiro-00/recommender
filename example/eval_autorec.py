from rec.cf.autorec import AutoRec
from  rec.util.data_loader import MovieLens

movie_lens = MovieLens("ml-100k")
data = movie_lens.rating_to_np()
autorec = AutoRec(data.shape[1])
autorec.train(data)