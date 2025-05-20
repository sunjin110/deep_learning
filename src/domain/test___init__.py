import unittest
import domain
import numpy as np

class TestSoftmax(unittest.TestCase):
    def test_overflow(self):
        result = domain.softmax(np.array([1010, 1000, 990]))
        self.assertFalse(np.all(np.isnan(result)))

if __name__ == '__main__':
    unittest.main()
