import logging
import numpy as np
from rec.metrics.similarity import pearson

class NeighborhoodModel():
    """Neighborhood model"""

    log = logging.getLogger("LOG")
    similarity_algorithm = {"pearson":pearson}

    def __init__(self, similarity = "pearson"):
        self.similarity = self.similarity_algorithm[similarity]

    def fill(self, rating_matrix):
        result = rating_matrix.copy()
        for u_index, u_column in enumerate(rating_matrix):
            u_average = u_column[u_column!=-1].mean()
            '''
            TODO: If average_rating==-1, use global average
            '''
            similarity_sum = 0
            predicted = np.zeros(rating_matrix.shape[1])
            for v_index, v_column in enumerate(rating_matrix):
                if u_index == v_index:
                    continue
                v_average = v_column[v_column!=-1].mean()
                similarity = self.similarity(u_column, v_column)
                predicted[v_column!=-1] += similarity * (v_average - v_column[v_column!=-1])
                similarity_sum += similarity
            predicted /= similarity_sum
            result[u_index][result[u_index]==-1] = u_average + predicted[result[u_index]==-1]

        return result

