import torch
from numpy.core.multiarray import dtype
from torch.autograd import Variable
from torch import FloatTensor, LongTensor

import abc, itertools, math
from numpy import prod

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import torchsample as ts
from torchsample.modules import ModuleTrainer

from torchsample.metrics import *

import time, random

from tqdm import trange

"""
TODO:
- Add (optional) bias node (subclass of HyperLayer chooses whether there's a bias, Hyperlayer handles it if given)
  Probably easiest to add some dimensions to the input tensor... let the hypernetwork figure out how much of a bias it needs.
- Proper experiment (CIFAR?). No convolutions in hypernetwork, only dense (and pooling). HyperLayer should discover structure, so that
  we can claim it is inferring structure from the data.


"""
def fi(indices, shape):
    """
    Returns the single index of the entry indicated by the given index-tuple, after a tensor (of the given shape) is
    flattened into a vector (by t.view(-1))

    :param indices:
    :param shape:
    :return:
    """
    batchsize, rank = indices.size()

    res = LongTensor(batchsize).fill_(0)
    for i in range(rank):
        prod = LongTensor(batchsize).fill_(1)
        for j in range(i + 1, len(shape)):
            prod *= shape[j]

        res += prod * indices[:, i]

    return res

def prod(tuple):
    result = 1

    for v in tuple:
        result *= v

    return result

def flatten_indices(indices, in_shape, out_shape):
    """
    Turns a n NxK matrix of N index-tuples for a tensor T of rank K into an Nx2 matrix M of index-tuples for a _matrix_
    that is created by flattening the first 'in_shape' dimensions into the vertical dimension of M and the remaining
    dimensions in the the horizontal dimension of M.

    :param indices: Variable containing long tensor  tensor
    :param in_rank:
    :return: A matrix of size N by 2. The resultant matrix is a LongTensor, but _not_ an autograd Variable.
    """
    batchsize, n, rank = indices.size()
    inrank = len(in_shape)

    result = LongTensor(batchsize, n, 2)

    for row in range(n):
        result[:, row, 0] = fi(indices[:, row, 0:inrank].data, in_shape)
        result[:, row, 1] = fi(indices[:, row, inrank:rank].data, out_shape)

    return result, (prod(in_shape), prod(out_shape))


def cache(indices, vals):
    """
    Store the parameters of the sparse matrix in a dictionary for easy computation of W * x
    """

    batchsize, n, _ = indices.size()

    result = [None] * batchsize

    for b in range(batchsize):

        result[b] = {}

        for row in range(n):
            i, j = indices[b, row, :]

            if i not in result[b]:
                result[b][i] = []

            result[b][i].append( (j, vals[b, row]) )

    return result


def discretize(ind, val):
    """
    Takes the output of a hypernetwork (rel-valued indices and corresponding values) and turns it into a list of
    integer indices, by "distributing" the values to the nearest neighboring integer indices.

    NB: the returned ints is not a Variable (just a plain LongTensor). autograd of the real valued indices passes
    through the values alone, not the integer indices used to instantiate the sparse matrix.

    :param ind: A Variable containing a matrix of N by K, where K is the number of indices.
    :param val: A Variable containing a vector of length N containing the values corresponding to the given indices
    :return: a triple (ints, props, vals). ints is an N*2^K by K matrix representing the N*2^K integer index-tuples that can
        be made by flooring or ceiling the indices in 'ind'. 'props' is a vector of length N*2^K, which indicates how
        much of the original value each integer index-tuple receives (based on the distance to the real-valued
        index-tuple). vals is vector of length N*2^K, containing the value of the corresponding real-valued index-tuple
        (ie. vals just repeats each value in the input 'val' 2^K times).
    """
    # ind = Variable(torch.rand(5, 2) * 20.0)
    # val = Variable(torch.rand(5) * 3.0)

    batchsize, n, rank = ind.size()

    # ints is the same size as ind, but for every index-tuple in ind, we add an extra axis containing the 2^rank
    # integerized index-tuples we can make from that one real-valued index-tuple
    ints = Variable(torch.FloatTensor(batchsize, n, 2 ** rank, rank))

    # produce all possible integerized index-tuples
    for row in range(ind.size()[1]):
        for t, bools in enumerate(itertools.product([True, False], repeat=ind.size()[1])):

            for col, bool in enumerate(bools):
                r = ind[:, row, col]
                ints[:, row, t, col] = torch.floor(r) if bool else torch.ceil(r)

    # compute the proportion of the value each integer index tuple receives
    diffs = ints - torch.unsqueeze(ind, 2)
    abs = torch.abs(diffs)
    props = 1.0 - abs
    props = torch.prod(props, 3)

    # props is batchsize x n x 2^rank, giving a weight to each neighboring integer index-tuple

    # repeat eaxh value 2^k times, so it matches the new indices
    val = torch.unsqueeze(val, 2).expand_as(props).contiguous()

    # 'Unroll' the ints tensor into a long list of integer index tuples (ie. a matrix of n*2^rank by rank for each
    # instance in the batch) ...
    ints = ints.view(batchsize, -1, rank, 1).squeeze(3)

    # ... and reshape the props and vals the same way
    props = props.view(batchsize, -1)
    val = val.view(batchsize, -1)

    return ints.long(), props, val

class HyperLayer(nn.Module):
    """
        Abstract class for the hyperlayer. Implement by defining a hypernetwork, and returning it from the hyper method.
    """

    @abc.abstractmethod
    def hyper(self, input):
        """
            Returns the hypernetwork. This network should take the same input as the hyperlayer itself
            and output a pair (L, V), with L a matrix of k by R (with R the rank of W) and a vector V of length k.
        """
        return

    def __init__(self, in_rank, out_shape, k):

        super(HyperLayer, self).__init__()

        self.in_rank = in_rank
        self.out_shape = out_shape # without batch dimension

        self.weights_rank = in_rank + len(out_shape) # implied rank of W

    def forward(self, input):

        batchsize = input.size()[0]

        real_indices, real_values = self.hyper(input)

        # NB: due to batching, real_indices has shape batchsize x K x rank(W)
        #     real_values has shape batchsize x K

        # turn the real values into integers in a differentiable way
        indices, props, values = discretize(real_indices, real_values)

        # translate tensor indices to matrix indices
        mindices, mshape = flatten_indices(indices, input.size()[1:], self.out_shape)

        # NB: mindices is not an autograd Variable. The error-signal for the indices passes to the hypernetwork
        # through 'values', which are a function of both the real_indices and the real_values.

        # Create the sparse weight tensor

        # -- Turns out we don't have autograd over sparse tensors yet (let alone over the constructor arguments). For
        #    now, we'll do a slow, naive multiplication.

        # Group the indices by the first index
        w = cache(mindices, props * values)

        x_flat = input.view(batchsize, -1)

        ly = prod(self.out_shape)
        y_flat = Variable(FloatTensor(batchsize, ly))

        for b in range(batchsize):
            for i in range(ly):
                y_flat[b, i] = sum([val * x_flat[b, j] for (j, val) in w[b][i] ])

        y_shape = [batchsize]
        y_shape.extend(self.out_shape)

        y = y_flat.view(y_shape) # reshape y into a proper tensor

        # if random.random() < 0.001:
        #    print(real_indices, real_values)

        return y

class SimpleHyperLayer(HyperLayer):
    """
    Simple function from 2-vector to a 2-vector, no bias.
    """


    def __init__(self):
        super().__init__(in_rank=1, out_shape=(2,), k=2)

        # hypernetwork
        self.hyp = nn.Sequential(
            nn.Linear(2,6),
            nn.Sigmoid(),
        )

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        res = self.hyp.forward(input)
        # res has shape batch_size x 6

        ind = res[:, 0:4]
        val = res[:, 4:6]

        return torch.unsqueeze(ind, 2).contiguous().view(-1, 2, 2), val

class Net(nn.Module):
    """
    The network containing the hyperlayers
    """

    def __init__(self):
        super(Net, self).__init__()
        self.hyper = SimpleHyperLayer()

    def forward(self, x):

        return self.hyper(x)

if __name__ == '__main__':

    torch.manual_seed(1)

    model = Net()

    # x = Variable(torch.rand((3, 2,)))
    # print(x)
    #
    # y = model(x)
    # print(y)


    N = 1000
    B = 50

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for i in trange(N):
        x = Variable( torch.rand((B, 2)) )

        optimizer.zero_grad()

        y = model(x)
        loss = criterion(y, x) # compute the loss
        loss.backward()        # compute the gradients
        optimizer.step()

    for i in range(20):
        x = Variable( torch.rand((B, 2,)) )
        y = model(x)

        print('diff', torch.abs(x - y))

        print('********')
