import torch
from numpy.core.multiarray import dtype
from torch.autograd import Variable
from torch.nn import Parameter
from torch import FloatTensor, LongTensor

import abc, itertools, math, time, random
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

from enum import Enum

from tqdm import trange

from hyper import Bias, Flatten, fi, flatten_indices, sort

from scipy.stats import multivariate_normal

"""
Sampling version of the ASH layer.
"""

def prod(tuple):
    result = 1

    for v in tuple:
        result *= v

    return result

def discretize(means, sigmas, vals, d, w_shape):
    """
    Takes the output of a hypernetwork (a set of multivariate normal distributions, combined with values) and turns it
    into a list of integer indices, by "distributing" each value/MVN to likely integer indices

    NB: the returned ints is not a Variable (just a plain LongTensor). autograd of the real valued indices passes
    through the values alone, not the integer indices used to instantiate the sparse matrix.

    :param means: A Variable containing a matrix of N by K, where K is the number of indices. Each row represents the
        mean of an
    :param: sigmas: A Variable containing a matix of
    :param val: A Variable containing a vector of length N containing the values corresponding to the given indices
    :return: a triple (ints, props, vals). ints is an N*2^K by K matrix representing the N*2^K integer index-tuples that can
        be made by flooring or ceiling the indices in 'ind'. 'props' is a vector of length N*2^K, which indicates how
        much of the original value each integer index-tuple receives (based on the distance to the real-valued
        index-tuple). vals is vector of length N*2^K, containing the value of the corresponding real-valued index-tuple
        (ie. vals just repeats each value in the input 'val' 2^K times).
    """
    batchsize, n, rank = means.size()

    # ints is the same size as means, but for every index-tuple in ind, we add an extra axis containing the d
    # index-tuples we sample for that one MVN
    ints = sample(means, sigmas, d)

    # Probability for each point (pdf normalized over the d points sampled from each MVN)
    probs = densities(means, sigmas, ints)

    # -- normalize
    sums = torch.sum(probs, dim=2, keepdim=True).expand_as(probs)
    probs = probs/sums

    # clip to legal index tuples
    ints = torch.round(ints).long()

    lower = Variable(torch.zeros(ints.size()).long())
    upper = Variable(LongTensor(w_shape))
    upper = upper - 1
    upper = upper.unsqueeze(0).unsqueeze(0)
    upper = upper.expand_as(ints)

    ints = torch.max(ints, lower)
    ints = torch.min(ints, upper)

    # repeat each value d times, so it matches the new indices
    vals = torch.unsqueeze(vals, 2).expand_as(probs).contiguous()

    # 'Unroll' the ints tensor into a long list of integer index tuples (ie. a matrix of n*d by rank for each
    # instance in the batch) ...
    ints = ints.view(batchsize, -1, rank, 1).squeeze(3)

    # ... and reshape the probs and vals the same way
    probs = probs.view(batchsize, -1)
    vals = vals.view(batchsize, -1)

    return ints, probs, vals

def sample(means, sigmas, d):
    """
    Returns d samples for each mean/sigma given
    :param means:
    :param sigmas:
    :return:
    """
    batchsize, n, rank = means.size()

    # Random epsilon input.
    result = Variable(torch.randn(batchsize, n, d, rank), requires_grad=True)

    # multiply by sigma

    sigmas = sigmas.unsqueeze(2).unsqueeze(2)
    result = result * sigmas.expand_as(result)

    # add the means
    means = means.unsqueeze(2)
    result = result + means.expand_as(result)

    return result

def densities(means, sigmas, points):
    """
    Compute the PDFs of the points under the given MVNs

    :param means:
    :param sigmas:
    :param points:
    :return:
    """

    # n: number of MVNs
    # d: number of points per MVN
    # rank: dim of points

    batchsize, n, d, rank = points.size()

    means = means.unsqueeze(2).expand_as(points)

    points = points - means

    # Compute dot products for all points
    # -- unroll the batch/n dimensions
    points = points.view(-1, 1, rank, 1).squeeze(3)
    # -- dot prod
    products = torch.bmm(points, points.transpose(1,2))
    # -- reconstruct shape
    products = products.view(batchsize, n, d)

    sigmas = sigmas.unsqueeze(2).expand_as(products)
    sigmas = torch.pow(sigmas, 2)

    den = torch.sqrt(Variable(FloatTensor([(2.0 * math.pi) ** rank])) * torch.pow(sigmas, rank))

    num = torch.exp(- products * (1.0/(2.0*sigmas)))

    return num/den


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

    def __init__(self, in_rank, out_shape, d=7, bias_type=Bias.DENSE):

        super(HyperLayer, self).__init__()

        self.in_rank = in_rank
        self.out_shape = out_shape # without batch dimension

        self.weights_rank = in_rank + len(out_shape) # implied rank of W

        self.bias_type = bias_type

        self.d = d

    def forward(self, input):

        batchsize = input.size()[0]
        w_size =  LongTensor(list(self.out_shape) + list(input.size()[1:]))

        ### Compute and unpack output of hypernetwork

        if self.bias_type == Bias.NONE:
            means, sigmas, values = self.hyper(input)
        if self.bias_type == Bias.DENSE:
            means, sigmas, values, bias = self.hyper(input)
        if self.bias_type == Bias.SPARSE:
            means, sigmas, values, bias_means, bias_sigmas, bias_values = self.hyper(input)

        # NB: due to batching, real_indices has shape batchsize x K x rank(W)
        #     real_values has shape batchsize x K

        # turn the real values into integers in a differentiable way
        indices, props, values = discretize(means, sigmas, values, self.d, w_size)
        values = values * props

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

        # print('<>', real_indices, real_values)
        # print('||', mindices, values)

        for b in range(batchsize):
            r_start = 0
            r_end = 0

            while r_end < mindices.size()[1]:

                while r_end < mindices.size()[1] and mindices[b, r_start, 0] == mindices[b, r_end, 0]:
                    r_end += 1

                i = mindices[b, r_start, 0]
                ixs = mindices[b, r_start:r_end, 1]

                y_flat[b, i] = torch.dot(values[b, r_start:r_end], x_flat[b, :][ixs])

                r_start = r_end


        y_shape = [batchsize]
        y_shape.extend(self.out_shape)

        y = y_flat.view(y_shape) # reshape y into a tensor

        ### Handle the bias
        if self.bias_type == Bias.DENSE:
            y = y + bias
        if self.bias_type == Bias.SPARSE: # not implemented yet
           bias = None

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

class ASHLayer(HyperLayer):
    """
    Hyperlayer with arbitrary (fixed) in/out shape. Uses simple dense hypernetwork
    """

    def __init__(self, in_shape, out_shape, k, d=3, hidden=256):
        super().__init__(in_rank=1, out_shape=out_shape, d=d, bias_type=Bias.NONE)

        self.k = k
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.w_rank = len(in_shape) + len(out_shape)

        # hypernetwork
        self.hyp = nn.Sequential(
            Flatten(),
            nn.Linear(prod(in_shape), hidden),
            nn.ReLU(),
            nn.Linear(hidden, (self.w_rank + 2) * k),
        )

        # self.bias = Parameter(torch.zeros(out_shape))

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        res = self.hyp.forward(input)
        # res has shape batch_size x 6

        means = nn.functional.sigmoid(res[:, 0:self.k * self.w_rank])
        means = means.unsqueeze(2).contiguous().view(-1, self.k, self.w_rank)

        ## expand the indices to the range [0, max]

        # Limits for each of the w_rank indices
        s = Variable(FloatTensor(list(self.out_shape) + list(input.size())[1:]).contiguous())
        s = s - 1
        s = s.unsqueeze(0).unsqueeze(0)
        s = s.expand_as(means)

        means = means * s

        sigmas = nn.functional.softplus(res[:, self.k * self.w_rank : self.k * self.w_rank + self.k])
        values = res[:, self.k * self.w_rank + self.k : ]

        return means, sigmas, values

class SimpleASHLayer(HyperLayer):
    """
    Simple layer for the 2D map
    """

    def __init__(self, d=3):
        super().__init__(in_rank=1, out_shape=(2,), d=d, bias_type=Bias.NONE)

        self.means = Parameter(torch.rand((1, 2, 2)))
        self.sigmas = Parameter(torch.rand(1, 2))
        self.values = Parameter(torch.rand(1, 2))

        # self.means = Parameter(FloatTensor( [[[0.01,0.01], [1.01,1.01]]] ))
        # self.sigmas = Parameter(FloatTensor( [[0.1,0.1]] ))
        # self.values = Parameter(FloatTensor( [[0.9,0.9]] ))

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        return self.means, self.sigmas, self.values


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

def pdf(point, mean, sigma):
    """
    For testing the implementation of densities()
    """
    x, y = point.data
    a, b = mean.data
    s = sigma.data * sigma.data

    return multivariate_normal.pdf([x, y], mean=[a, b], cov = [[s[0], 0],[0, s[0]]])

if __name__ == '__main__':

    # TODO Make this a unit test
    mus = Variable(FloatTensor([[[10.0, 10.0], [20.0, 20.0]], [[100.0, 100.0], [200.0, 200.0]]]))
    sigs = Variable(FloatTensor([[0.1, 0.01], [10.0, 0.1]]))
    res = sample(mus, sigs, 3)
    res = torch.round(res).long()

    # print(res[0, 0, 2, :]) # 10,10 0.1
    # print(res[0, 1, 2, :]) # 20,20 0.01
    # print(res[1, 0, 2, :]) # 100,100 10.0
    # print(res[1, 1, 2, :]) # 200,200 0.1

    points = Variable(FloatTensor([[[
            [11.0, 11.0], [12.0, 12.0], [9.0, 10.0]
        ], [
            [21.0, 20.0], [20.0, 20.0], [19.0, 19.0]
        ]], [[
            [100.0, 100.0], [105.0, 100.0], [99.0, 99.0]
        ], [
            [201.0, 201.0], [202.0, 202.0], [199.0, 200.0]
        ]]]))

    actual = densities(mus, sigs, points)

    # for b in range(2):
    #     for n in range(2):
    #         for d in range(3):
    #             print(pdf(points[b, n, d, :], mus[b, n, :], sigs[b, n]), actual[b, n, d].data[0])
    #

    print(actual)
    sums = torch.sum(actual, dim=2).unsqueeze(2).expand_as(actual)
    print(actual/sums)
