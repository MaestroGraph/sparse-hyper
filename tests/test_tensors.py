import _context

import unittest
import torch
from torch.autograd import Variable
#import sparse.layers.densities

import tensors, time


def sample(nindices=2*256+2*8, size=(256, 256), var=1.0):
    assert len(size) == 2

    indices = (torch.rand(nindices, 2) * torch.tensor(size)[None, :].float()).long()
    values = torch.randn(nindices) * var

    return indices, values

class TestTensors(unittest.TestCase):

    def test_sum(self):
        size = (5, 5)

        # create a batch of sparse matrices
        samples = [sample(nindices=3, size=size) for _ in range(3)]
        indices, values = [s[0][None, :, :] for s in samples], [s[1][None, :] for s in samples]

        indices, values = torch.cat(indices, dim=0), torch.cat(values, dim=0)

        print(indices)
        print(values)
        print('res', tensors.sum(indices, values, size))

    def test_log_softmax(self):

        size = (5, 5)

        # create a batch of sparse matrices
        samples = [sample(nindices=3, size=size) for _ in range(3)]
        indices, values = [s[0][None, :, :] for s in samples], [s[1][None, :] for s in samples]

        indices, values = torch.cat(indices, dim=0), torch.cat(values, dim=0)

        print('res', tensors.logsoftmax(indices, values, size, method='naive').exp())
        print('res', tensors.logsoftmax(indices, values, size, method='iteration').exp())

    def test(self):

        a = Variable(torch.randn(1), requires_grad=True)
        x = Variable(torch.randn(15000, 15000))

        x = x * a
        x = x / 2

        loss = x.sum()

        loss.backward()
        time.sleep(600)
