import unittest
import hyper
import torch

class SimpleTests(unittest.TestCase):

    def test_fi(self):
        input = torch.LongTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        expected = torch.LongTensor([0, 1, 2, 3])

        actual = hyper.fi(input, (2,2))

        # TODO figure out how unit tests are supposed to work
if __name__ == '__main__':
    unittest.main()
