import torch
from numpy.core.multiarray import dtype
from torch.autograd import Variable

import abc, itertools, math
from numpy import prod

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.sparse import FloatTensor

import torchvision
import torchvision.transforms as transforms

import torchsample as ts
from torchsample.modules import ModuleTrainer

from torchsample.metrics import *

import time

EPOCHS = 350
BATCH_SIZE = 64
GPU = False

SMALL = 96
BIG = 192

# Set up the dataset

normalize = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=normalize)

trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=normalize)

testloader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def fi(indices, shape):
    """
    Returns the single index of the entry indicated by the given index-tuple, after a tensor (of the given shape) is
    flattened into a vector (by t.view(-1))

    :param indices:
    :param shape:
    :return:
    """
    res = 0
    for i in range(len(indices)):
        prod = 1
        for j in range(i + 1, len(shape)):
            prod *= shape[j]

        res += prod * indices[i]

    return res

def flatten_indices(indices, in_shape, out_shape):
    """
    Turns a n NxK matrix of N index-tuples for a tensor T of rank K into an Nx2 matrix M of index-tuples for a _matrix_
    that is created by flattening the first 'in_shape' dimensions into the vertical dimension of M and the remaining
    dimensions in the the horizontal dimension of M.

    :param indices:
    :param in_rank:
    :return: A matrix of size N by 2. The resultant matrix is a LongTensor, but _not_ an autograd Variable.
    """
    n, k = indices.size()
    inrank = len(in_shape.size())

    result = torch.LongTensor((n, 2))

    for row in range(n):
        result[row, 0] = fi(indices[row, 0:inrank], in_shape)
        result[row, 1] = fi(indices[row, inrank:k], out_shape)

    return result, (torch.prod(in_shape), torch.prod(out_shape))

def discretize(ind, val):
    """
    Takes the output of a hypernetwork (rel-valued indices and corresponding values) and turns it into a list of
    integer indices, by "distributing" the values to the nearest neighboring integer indices.

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

    n, k = ind.size()

    # ints is the same size as ind, but for every index-tuple in ind, we add an extra axis containing the 2^k
    # integerized index-tuples we can make from that one real-valued index-tuple
    ints = Variable(torch.zeros(n, 2 ** k, k))

    # produce all possible integerized index-tuples
    for row in range(ind.size()[0]):
        for t, bools in enumerate(itertools.product([True, False], repeat=ind.size()[1])):

            for col, bool in enumerate(bools):
                r = ind[row, col]
                ints[row, t, col] = torch.floor(r) if bool else torch.ceil(r)

    # compute the proportion of the value each integer index tuple receives
    diffs = ints - torch.unsqueeze(ind, 1)
    abs = torch.abs(diffs)
    props = 1.0 - abs
    props = torch.prod(props, 2)

    val = torch.unsqueeze(val, 1).expand_as(props).contiguous()

    # flatten the ints tensor into a long list of integer index tuples (ie. a matrix of n*2^k by k) ...
    ints = ints.view(-1, k, 1).squeeze()
    # ... and reshape the props and vals the same way
    props = props.view(-1)
    val = val.view(-1)

    return ints, props, val

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
        self.out_shape = out_shape

        self.weights_rank = in_rank + len(out_shape) # implied rank of W

    def forward(self, input):

        real_indices, real_values = self.hyper(input)

        # turn the real values into integers in a differentiable way
        indices, props, values = discretize(real_indices, real_values)

        # translate tensor indices to matrix indices
        mindices, mshape = flatten_indices(indices, input.size(), self.out_shape)

        # NB: mindices is not an autograd Variable. The error-signal for the indices passes through to the hypernetwork
        # through the new values, which are a function of both the real_indices and the real_values.

        # Create the sparse weight tensor
        weights = Variable(FloatTensor(mindices.transpose(), props * values, torch.Size(mshape)))

        return weights.mm(input)

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

        ind = res[0:4]
        val = res[4:6]

        return torch.unsqueeze(ind).view(2, 2), val

class Net(nn.Module):
    """
    The network containing the hyperlayers
    """

    def __init__(self):
        super(Net, self).__init__()
        self.hyper = SimpleHyperLayer()

    def forward(self, x):

        return self.hyper(x)

model = Net()

x = torch.Tensor([1,1])

print(x)
print(model(x))

# N = 10000
# EPOCHS = 5
# data = torch.rand((N, 2))
#
# trainer = ModuleTrainer(model)
#
# trainer.compile(
#     loss=nn.MSELoss(),
#     optimizer='adam',
#     metrics=[CategoricalAccuracy()])
#
# trainer.fit(data, data,
#     nb_epoch=EPOCHS,
#     batch_size=128,
#     verbose=1)