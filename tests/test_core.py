import unittest
from tensorwich.core import TensorWich

class TestTensorWich(unittest.TestCase):

    def setUp(self):
        self.tensor = TensorWich()

    def test_add(self):
        result = self.tensor.add(1, 2)
        self.assertEqual(result, 3)

    def test_subtract(self):
        result = self.tensor.subtract(5, 3)
        self.assertEqual(result, 2)

    def test_multiply(self):
        result = self.tensor.multiply(3, 4)
        self.assertEqual(result, 12)

if __name__ == '__main__':
    unittest.main()