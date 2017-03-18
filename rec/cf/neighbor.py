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
        for u_index, u_row in enumerate(rating_matrix):
            u_mean = u_row[u_row!=-1].mean()
            '''
            TODO: If average_rating==-1, use global average
            '''
            similarity_sum = 0
            predicted = np.zeros(rating_matrix.shape[1])
            for v_index, v_row in enumerate(rating_matrix):
                if u_index == v_index:
                    continue
                v_average = v_row[v_row!=-1].mean()
                try:
                    similarity = self.similarity(u_row, v_row)
                    predicted[v_row!=-1] += similarity * (v_average - v_row[v_row!=-1])
                    similarity_sum += similarity
                except:
                    pass
            if similarity_sum != 0:
                predicted /= similarity_sum
            result[u_index][result[u_index]==-1] = u_mean + predicted[result[u_index]==-1]

        return result

