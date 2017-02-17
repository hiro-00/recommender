import logging

class NeighborhoodModel():
    """Neighborhood model"""

    log = logging.getLogger("LOG")

    def __init__(self, similarity = "pearson"):
        self.similarity = similarity

    def predict(self, rating_matrix):
        for user in rating_matrix:
            average_rating = user[user!=-1].mean()

