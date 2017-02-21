import pandas as pd
import numpy as np
import os

class MovieLens():
    def __init__(self, dataset):
        self.path = os.path.dirname(__file__) + "/../../data/" + dataset
    def rating_to_np(self):
        data = []
        user_id_num = 0
        movie_id_num = 0
        with open(self.path + "/ratings.dat") as file:
             for line in file:
                user_id, movie_id, rating, timestamp = [int(d) for d in line.split("::")]
                data.append((user_id - 1, movie_id - 1, rating))
                user_id_num = max(user_id_num, user_id)
                movie_id_num = max(movie_id_num, movie_id)
        result = np.full((user_id_num, movie_id_num), -1)
        for d in data:
            result[d[0]][d[1]] = d[2]
        return result

    def rating_to_sparse(self):
        pass


test = MovieLens("ml-100k")
rating = test.rating_to_np()
rating[rating != -1]=1
rating[rating == -1]=0

print(rating)
