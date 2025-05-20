import unittest
from domain.model import sum_squared_error, cross_entropy_error
import numpy as np

class TestSumSquaredError(unittest.TestCase):
    def test(self):
        # 2を正解とする
        t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ]

        # 例1: 2の確率が最も高い場合
        y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
        result = sum_squared_error(np.array(y), np.array(t))
        print("example1: {0}".format(result))

        # 例2
        y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
        result = sum_squared_error(np.array(y), np.array(t))
        print("example2: {0}".format(result))

class TestCrossEntropyError(unittest.TestCase):
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ]

    def test(self):
        y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

        result = cross_entropy_error(np.array(y), np.array(self.t))
        print("example1: {0}".format(result))

        y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
        result = cross_entropy_error(np.array(y), np.array(self.t))
        print("example2: {0}".format(result))

if __name__ == '__main__':
    unittest.main()
