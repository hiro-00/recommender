import unittest
import logging
import numpy as np
from rec.cf.neighbor import NeighborhoodModel

class TestNeighbor(unittest.TestCase):
    def setUp(self):
        self.rating = np.array([[1,2,3],
                             [0,-1,1]])
        self.model = NeighborhoodModel()

    def test_predict(self):
        self.model.predict(self.rating)