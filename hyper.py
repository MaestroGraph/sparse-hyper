import torch
from numpy.core.multiarray import dtype
from torch.autograd import Variable
from torch.nn import Parameter
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
from util import *

import time, random

from enum import Enum

from tqdm import trange

"""
TODO:
- CIFAR, identity function
- Proper experiment (CIFAR?). No convolutions in hypernetwork, only dense (and pooling). HyperLayer should discover sparse structure, so that
  we can claim it is inferring structure from the data.
"""

class Bias(Enum):
    """

    """

    # No bias is used.
    NONE = 1

    # The bias is returned as a single dense tensor of floats.
    DENSE = 2

    # The bias is returnd in sparse format, in the same way as the weight matrix is.
    SPARSE = 3

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

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
    outrank = len(out_shape)

    result = LongTensor(batchsize, n, 2)

    for row in range(n):
        result[:, row, 0] = fi(indices[:, row, 0:outrank].data, out_shape)   # i index of the weight matrix
        result[:, row, 1] = fi(indices[:, row, outrank:rank].data, in_shape) # j index

    return result, (prod(out_shape), prod(in_shape))

def sort(indices, vals):
    """

    :param indices:
    :return:
    """
    batchsize, n, _ = indices.size()

    inew = LongTensor(indices.size())
    vnew = Variable(torch.zeros(vals.size()))

    for b in range(batchsize):

        _, ixs = torch.sort(indices[b, :, 0])
        inew[b, :, :] = indices[b, :, :][ixs]
        vnew[b, :] = vals[b, :][ixs]

    return inew, vnew

# def cache(indices, vals):
#     """
#     Store the parameters of the sparse matrix in a dictionary for easy computation of W * x
#
#     indices is assumed to be sorted by row.
#
#     """
#
#     batchsize, n, _ = indices.size()
#
#     # contains the i index for each index-tuple
#     rows = FloatTensor((batchsize, n))
#
#     # contains the j index and corresponding value for each tuple
#     values = FloatTensor((batchsize, n, 2))
#
#     for b in range(batchsize):
#
#         rows[b] = []
#         values[b] = []
#
#         for row in range(n):
#             i, j = indices[b, row, :]
#
#
#
#     return rows, values


def discretize(ind, val):
    """
    Takes the output of a hypernetwork (real-valued indices and corresponding values) and turns it into a list of
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
    for row in range(n):
        for t, bools in enumerate(itertools.product([True, False], repeat=rank)):

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

    def __init__(self, in_rank, out_shape, bias_type=Bias.DENSE):

        super(HyperLayer, self).__init__()

        self.in_rank = in_rank
        self.out_shape = out_shape # without batch dimension

        self.weights_rank = in_rank + len(out_shape) # implied rank of W

        self.bias_type = bias_type

    def forward(self, input):

        batchsize = input.size()[0]

        ### Compute and unpack ouput of hypernetwork

        if self.bias_type == Bias.NONE:
            real_indices, real_values = self.hyper(input)
        if self.bias_type == Bias.DENSE:
            real_indices, real_values, bias = self.hyper(input)
        if self.bias_type == Bias.SPARSE:
            real_indices, real_values, bias_indices, bias_values = self.hyper(input)

        # NB: due to batching, real_indices has shape batchsize x K x rank(W)
        #     real_values has shape batchsize x K

        # turn the real values into integers in a differentiable way
        indices, props, values = discretize(real_indices, real_values)

        # translate tensor indices to matrix indices
        mindices, _ = flatten_indices(indices, input.size()[1:], self.out_shape)

        # NB: mindices is not an autograd Variable. The error-signal for the indices passes to the hypernetwork
        #     through 'values', which are a function of both the real_indices and the real_values.

        ### Create the sparse weight tensor

        # -- Turns out we don't have autograd over sparse tensors yet (let alone over the constructor arguments). For
        #    now, we'll do a slow, naive multiplication.

        x_flat = input.view(batchsize, -1)

        ly = prod(self.out_shape)
        y_flat = Variable(torch.zeros((batchsize, ly)))


        mindices, values = sort(mindices, values)

        for b in range(batchsize):
            r_start = 0
            r_end = 0

            while mindices[b, r_start, 0] == mindices[b, r_end, 0]:
                r_end += 1

            i = mindices[b, r_start, 0]
            ixs = mindices[b, r_start:r_end, 1]
            y_flat[b, i] = torch.dot(values[b, r_start:r_end], x_flat[b, :][ixs])

        y_shape = [batchsize]
        y_shape.extend(self.out_shape)

        y = y_flat.view(y_shape) # reshape y into a tensor

        ### Handle the bias
        if self.bias_type == Bias.DENSE:
            y = y + bias
        if self.bias_type == Bias.SPARSE: # untested!
            bindices, bprops, bvalues = discretize(bias_indices, bias_values)
            vals = bprops * bvalues

            for b in range(batchsize):
                for row in range(bindices.size()[1]):
                    index = bindices[b, row, :]
                    y[index] += vals[b, row]

        return y

class SimpleHyperLayer(HyperLayer):
    """
    Simple function from 2-vector to a 2-vector, no bias.
    """


    def __init__(self):
        super().__init__(in_rank=1, out_shape=(2,), bias_type=Bias.DENSE)

        # hypernetwork
        self.hyp = nn.Sequential(
            nn.Linear(2,8),
            nn.Sigmoid(),
        )

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        res = self.hyp.forward(input)
        # res has shape batch_size x 6

        ind  = res[:, 0:4]
        val  = res[:, 4:6]
        bias = res[:, 6:8]

        return torch.unsqueeze(ind, 2).contiguous().view(-1, 2, 2), val, bias

class ImageHyperLayer(HyperLayer):
    """
    Function from one 3-tensor to another, with dense bias not learned from a hypernetwork
    """

    def __init__(self, in_shape, out_shape, k, poolsize=4, hidden=256):
        super().__init__(in_rank=3, out_shape=out_shape, bias_type=Bias.DENSE)

        self.k = k

        c, x, y = in_shape
        flat_size = int(x/poolsize) * int(y/poolsize) * c

        # hypernetwork
        self.hyp = nn.Sequential(
            nn.MaxPool2d(kernel_size=poolsize, stride=poolsize),
            Flatten(),
            nn.Linear(flat_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, k * 6 + k)
        )

        self.bias = Parameter(torch.zeros(out_shape))

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        res = self.hyp.forward(input)
        # res has shape batch_size x 6

        ind = nn.functional.sigmoid(res[:, 0:self.k*6])
        ind = ind.unsqueeze(2).contiguous().view(-1, self.k, 6)

        ## expand the indices to the range [0, max]

        # Limits for each of the 6 indices
        s = Variable(FloatTensor(list(self.out_shape) + list(input.size())[1:] ).contiguous())
        s = s - 1
        s = s.unsqueeze(0).unsqueeze(0)
        s = s.expand_as(ind)

        ind = ind * s

        val = res[:, self.k*6:self.k*6 + self.k]

        return ind, val, self.bias

class SimpleNet(nn.Module):
    """
    The network containing the hyperlayers
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.hyper = SimpleHyperLayer()

    def forward(self, x):

        return self.hyper(x)

if __name__ == '__main__':

    torch.manual_seed(1)

    ### CIFAR Experiment
    EPOCHS = 10
    BATCH_SIZE = 32

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

    model = nn.Sequential(
        ImageHyperLayer((3, 32, 32),  (1, 4, 4), 20),
      # ImageHyperLayer((16, 16, 16), (32, 8, 8),   10),
      # ImageHyperLayer((32, 8, 8),   (64, 4, 4),   10),
        Flatten(),
        nn.Linear(16, 10),
        nn.Softmax()
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    tic()

    running_loss = 0.0

    for epoch in range(EPOCHS):
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i != 0 and i % 50== 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch+1, i + 1, running_loss / BATCH_SIZE))
                running_loss = 0.0

    print('Finished Training. Took {} seconds'.format(toc()))

    ### SIMPLE
    # x = Variable(torch.rand((3, 2,)))
    # print(x)
    #
    # y = model(x)
    # print(y)


    # N = 50000
    # B = 1
    #
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters())
    #
    # for i in trange(N):
    #     x = Variable( torch.rand((B, 2)) )
    #
    #     optimizer.zero_grad()
    #
    #     y = model(x)
    #     loss = criterion(y, x) # compute the loss
    #     loss.backward()        # compute the gradients
    #     optimizer.step()
    #
    # for i in range(20):
    #     x = Variable( torch.rand((3, 2,)) )
    #     y = model(x)
    #
    #     print('diff', torch.abs(x - y))
    #
    #     print('********')
