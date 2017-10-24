import torch
from numpy.core.multiarray import dtype
from torch.autograd import Variable
from torch.nn import Parameter
from torch import FloatTensor, LongTensor

import abc, itertools, math, types
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
import util

import sys
import time, random

from enum import Enum

from tqdm import trange

import hyper

# added to the sigmas to prevent NaN
EPSILON = 10e-10

"""

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


def flatten(input):
    return input.view(input.size(0), -1)

def fi(indices, shape, use_cuda=False):
    """
    Returns the single index of the entry indicated by the given index-tuple, after a tensor (of the given shape) is
    flattened into a vector (by t.view(-1))

    :param indices:
    :param shape:
    :return:
    """
    batchsize, rank = indices.size()

    res = torch.cuda.LongTensor(batchsize).fill_(0) if use_cuda else LongTensor(batchsize).fill_(0)

    for i in range(rank):
        prod = torch.cuda.LongTensor(batchsize).fill_(1) if use_cuda else LongTensor(batchsize).fill_(1)
        if use_cuda:
            prod = prod.cuda()

        for j in range(i + 1, len(shape)):
            prod *= shape[j]

        res += prod * indices[:, i]

    return res

def prod(tuple):
    result = 1

    for v in tuple:
        result *= v

    return result

def flatten_indices(indices, in_shape, out_shape, use_cuda=False):
    """
    Turns a n NxK matrix of N index-tuples for a tensor T of rank K into an Nx2 matrix M of index-tuples for a _matrix_
    that is created by flattening the first 'in_shape' dimensions into the vertical dimension of M and the remaining
    dimensions in the the horizontal dimension of M.

    :param indices: Long tensor
    :param in_rank:
    :return: A matrix of size N by 2. .
    """

    batchsize, n, rank = indices.size()

    inrank = len(in_shape)
    outrank = len(out_shape)

    result = torch.cuda.LongTensor(batchsize, n, 2) if use_cuda else LongTensor(batchsize, n, 2)

    for row in range(n):
        result[:, row, 0] = fi(indices[:, row, 0:outrank], out_shape, use_cuda)   # i index of the weight matrix
        result[:, row, 1] = fi(indices[:, row, outrank:rank], in_shape, use_cuda) # j index

    return result, (prod(out_shape), prod(in_shape))

def sort(indices, vals, use_cuda=False):
    """

    :param indices:
    :return:
    """
    batchsize, n, _ = indices.size()

    inew = torch.cuda.LongTensor(indices.size()) if use_cuda else LongTensor(indices.size())
    vnew = torch.cuda.FloatTensor(vals.size()) if use_cuda else FloatTensor(vals.size())

    vnew = Variable(vnew)

    for b in range(batchsize):

        _, ixs = torch.sort(indices[b, :, 0])
        inew[b, :, :] = indices[b, :, :][ixs]
        vnew[b, :] = vals[b, :][ixs]

    return inew, vnew

def densities(points, means, sigmas):
    """
    Compute the unnormalized PDFs of the points under the given MVNs

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

    num = torch.exp(- products * (1.0/(2.0*(sigmas + EPSILON))))

    return num

def discretize(means, sigmas, values, use_cuda = False):
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

    batchsize, n, rank = means.size()

    # ints is the same size as ind, but for every index-tuple in ind, we add an extra axis containing the 2^rank
    # integerized index-tuples we can make from that one real-valued index-tuple
    ints = torch.cuda.FloatTensor(batchsize, n, 2 ** rank, rank) if use_cuda else FloatTensor(batchsize, n, 2 ** rank, rank)
    ints = Variable(ints)

    # produce all integerized index-tuples that neighbor the means
    for row in range(n):
        for t, bools in enumerate(itertools.product([True, False], repeat=rank)):

            for col, bool in enumerate(bools):
                r = means[:, row, col]
                ints[:, row, t, col] = torch.floor(r) if bool else torch.ceil(r)

    # compute the proportion of the value each integer index tuple receives
    props = densities(ints, means, sigmas)
    # props is batchsize x K x 2^rank, giving a weight to each neighboring integer index-tuple
    # -- normalize
    sums = torch.sum(props, dim=2, keepdim=True).expand_as(props)
    props = props/sums

    # repeat each value 2^k times, so it matches the new indices
    val = torch.unsqueeze(values, 2).expand_as(props).contiguous()

    # 'Unroll' the ints tensor into a long list of integer index tuples (ie. a matrix of n*2^rank by rank for each
    # instance in the batch) ...
    ints = ints.view(batchsize, -1, rank, 1).squeeze(3)

    # ... and reshape the props and vals the same way
    props = props.view(batchsize, -1)
    val = val.view(batchsize, -1)

    return ints.data.long(), props, val

class HyperLayer(nn.Module):
    """
        Abstract class for the hyperlayer. Implement by defining a hypernetwork, and returning it from the hyper() method.
    """
    @abc.abstractmethod
    def hyper(self, input):
        """
            Returns the hypernetwork. This network should take the same input as the hyperlayer itself
            and output a pair (L, V), with L a matrix of k by R (with R the rank of W) and a vector V of length k.
        """
        return

    def cuda(self, device_id=None):
        self.use_cuda = True
        super().cuda(device_id)

    def __init__(self, in_rank, out_shape, bias_type=Bias.DENSE):

        super().__init__()

        self.use_cuda = False
        self.in_rank = in_rank
        self.out_shape = out_shape # without batch dimension

        self.weights_rank = in_rank + len(out_shape) # implied rank of W

        self.bias_type = bias_type

    def forward(self, input):

        batchsize = input.size()[0]

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
        indices, props, values = discretize(means, sigmas, values, self.use_cuda)
        values = values * props

        # translate tensor indices to matrix indices
        mindices, _ = flatten_indices(indices, input.size()[1:], self.out_shape, self.use_cuda)

        # NB: mindices is not an autograd Variable. The error-signal for the indices passes to the hypernetwork
        #     through 'values', which are a function of both the real_indices and the real_values.

        ### Create the sparse weight tensor

        # -- Turns out we don't have autograd over sparse tensors yet (let alone over the constructor arguments). For
        #    now, we'll do a slow, naive multiplication.

        x_flat = input.view(batchsize, -1)

        ly = prod(self.out_shape)

        y_flat = torch.cuda.FloatTensor(batchsize, ly) if self.use_cuda else FloatTensor(batchsize, ly)
        y_flat.fill_(0.0)
        y_flat = Variable(y_flat)

        mindices, values = sort(mindices, values, self.use_cuda)

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
        if self.bias_type == Bias.SPARSE: # untested!
            pass

        return y

    def initialize(self, in_shape, batch_size=64, iterations=250, lr=0.001, verbose=True):
        """
        We aim to initialize to a generally orthonormal weight-tensor. To be precise, the subtensors of any two output
        nodes y_i and y_j should be orthogonal, when flattened to vectors, and each such subtensor should have unit
        length when flattened to a vector. This intialization ensures that each output nodeis initialized to represent
        independent aspects of the input, and that the outputs and gradients do not blow up or fade away over successive
        layers.

        For a simple, generic training objective, we use the property that orthonormal matrices preserve the dot-product. We
        sample two vectors x1 and x2, with independently drawn standard-normally distributed elements, and compute their
        respective outputs y1 and y2. We then use (dot(x1, x2) - dot(y1, y2))^2 as our loss function and optimize with
        Adam.

        NB: For small batch sizes, the loss tends to go to inf/NaN

        :param: in_shape Either a tuple representing the input shape, or (if the input has variable shape) a generator
        that generates random input shapes according to the data distribution.
        :return:
        """

        from tensorboardX import SummaryWriter
        w = SummaryWriter()

        y_size = (batch_size, ) + self.out_shape

        optimizer = torch.optim.Adam(self.parameters(),lr=lr)

        print('Initializing')

        restart = True
        while restart:
            restart = False
            for i in (trange(iterations) if verbose else range(iterations)):
                optimizer.zero_grad()

                # sample an input shape (if we have a generator
                ins = in_shape.next() if isinstance(in_shape, types.GeneratorType) else in_shape
                x_size = (batch_size,) + ins

                x1, x2 = torch.randn(x_size), torch.randn(x_size)

                # normalize to unit tensors
                x1, x2 = util.norm(x1), util.norm(x2)

                if self.use_cuda:
                    x1, x2 = x1.cuda(), x2.cuda()
                x1, x2 = Variable(x1), Variable(x2)

                y1 = self.forward(x1)
                y2 = self.forward(x2)

                x1 = x1.view(batch_size, 1, -1)
                x2 = x2.view(batch_size, 1, -1)
                y1 = y1.view(batch_size, 1, -1)
                y2 = y2.view(batch_size, 1, -1)

                xnorm = torch.bmm(x1, x2.transpose(1, 2))
                ynorm = torch.bmm(y1, y2.transpose(1, 2))

                # print('xy',x1, x2, y1, y2)
                # print(xnorm)

                loss = torch.sum(torch.pow((xnorm - ynorm), 2)) / batch_size
                # print('LOSS', loss)

                if math.isnan(loss.data[0]) or math.isinf(loss.data[0]):
                    if verbose:
                        print('Infinite or NaN loss encountered, restarting with new parameters.')

                        # print('x1', x1)
                        # print('x2', x2)
                        # print(xnorm)
                        #
                        # print('y1', y1)
                        # print('y2', y2)
                        # print(ynorm)
                        #
                        # print('loss (per batch)', torch.pow((xnorm - ynorm), 2))

                        sys.exit(1)
                    def reset(t):
                        if hasattr(t, 'reset_parameters'):
                            t.reset_parameters()
                    self.apply(reset)

                    restart = True
                    break

                w.add_scalar('init/loss', loss.data[0], i)

                loss.backward()
                optimizer.step()

                # if i % 50 == 0:
                #     means, sigmas, values = self.hyper(Variable(torch.randn(x_size)))
                #     plt.clf()
                #     util.plot(means, sigmas, values)
                #
                #     plt.xlim((-1, 8))
                #     plt.ylim((-1, 8))
                #     plt.axis('equal')
                #
                #     plt.savefig('./init/means.{:06}.png'.format(i))



def split_out(res, input_size, output_size, gain=5.0):

    b, k, width = res.size()
    w_rank = width - 2

    means = nn.functional.sigmoid(res[:, :, 0:w_rank] * gain)
    means = means.unsqueeze(2).contiguous().view(-1, k, w_rank)

    ## expand the indices to the range [0, max]

    # Limits for each of the w_rank indices
    s = Variable(FloatTensor(list(output_size) + list(input_size)).contiguous())
    s = s - 1
    s = s.unsqueeze(0).unsqueeze(0)
    s = s.expand_as(means)

    means = means * s

    sigmas = nn.functional.softplus(res[:, :, w_rank:w_rank+1]).squeeze(2)
    values = res[:, :, w_rank+1:].squeeze(2)

    return means, sigmas, values

class DenseASHLayer(HyperLayer):
    """
    Hyperlayer with arbitrary (fixed) in/out shape. Uses simple dense hypernetwork
    """

    def __init__(self, in_shape, out_shape, k, hidden=256):
        super().__init__(in_rank=len(in_shape), out_shape=out_shape, bias_type=Bias.NONE)

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

        means, sigmas, values = split_out(res, input.size()[1:], self.out_shape)

        return means, sigmas, values


class ParamASHLayer(HyperLayer):
    """
    Hyperlayer with arbitrary (fixed) in/out shape. Uses simple dense hypernetwork
    """

    def __init__(self, in_shape, out_shape, k, hidden=256):
        super().__init__(in_rank=len(in_shape), out_shape=out_shape, bias_type=Bias.NONE)

        self.k = k
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.w_rank = len(in_shape) + len(out_shape)

        self.params = Parameter(torch.randn(k, self.w_rank + 2))

        # self.bias = Parameter(torch.zeros(out_shape))

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """
        batch_size = input.size()[0]

        # Replicate the parameters along the batch dimension
        res = self.params.unsqueeze(0).expand(batch_size, self.k, self.w_rank+2)

        means, sigmas, values = split_out(res, input.size()[1:], self.out_shape)

        return means, sigmas, values

class ImageCASHLayer(HyperLayer):
    """
    """

    def __init__(self, in_shape, out_shape, k, poolsize=4):
        super().__init__(in_rank=len(in_shape), out_shape=out_shape, bias_type=Bias.DENSE)

        self.k = k
        self.in_shape = in_shape
        self.out_shape = out_shape

        rep = 4*4*4*2

        self.w_rank = len(in_shape) + len(out_shape)

        c, x, y = in_shape
        flat_size = int(x/poolsize) * int(y/poolsize) * c

        # hypernetwork
        self.tohidden = nn.Sequential(
            nn.MaxPool2d(kernel_size=poolsize, stride=poolsize),
            Flatten(),
            nn.Linear(flat_size, int(k/rep)),
            nn.ReLU()
        )

        self.conv1da = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, stride=4)
        self.conv1db = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, stride=4)
        self.conv1dc = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, stride=4)

        self.conv2d = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(2, self.w_rank+2), stride=2)

        self.bias = nn.Sequential(
            nn.Linear(int(k/rep), hyper.prod(out_shape)),
        )

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        insize = input.size()

        hidden = self.tohidden(input)

        res = hidden

        res = res.unsqueeze(1)
        res = nn.functional.relu(self.conv1da(res))
        res = res.squeeze(1)

        res = res.unsqueeze(1)
        res = nn.functional.relu(self.conv1db(res))
        res = res.squeeze(1)

        res = res.unsqueeze(1)
        res = nn.functional.relu(self.conv1dc(res))
        res = res.squeeze(1)

        res = res.unsqueeze(1).unsqueeze(3)
        res = nn.functional.relu(self.conv2d(res))
        res = res.squeeze(1)

        means, sigmas, values = split_out(res, input.size()[1:], self.out_shape)

        bias = self.bias(hidden)
        bias = bias.view((-1, ) + self.out_shape)

        return means, sigmas, values, bias

class CASHLayer(HyperLayer):
    """

    """
    def __init__(self, in_shape, out_shape, k, poolsize=4, deconvs=2, gain=7.0):
        """
        :param in_shape:
        :param out_shape:
        :param k: How many index tuples to generate. If this is not divisible by 2^deconvs, you'll get the next biggest
        number that is.
        :param poolsize:
        :param deconvs: How many deconv layers to use to generate the tuples from the hidden layer
        """
        super().__init__(in_rank=len(in_shape), out_shape=out_shape, bias_type=Bias.DENSE)

        class NoActivation(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, input):
                return input

        self.activation = NoActivation()

        self.k = k
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.gain = gain

        self.w_rank = len(in_shape) + len(out_shape)

        hidden = int(math.ceil(k/2**deconvs))

        if len(in_shape) == 1:
            x = in_shape[0]
            flat_size = int(x / poolsize)
            self.pool = nn.AvgPool1d(kernel_size=poolsize, stride=poolsize)
        elif len(in_shape) == 2:
            x, y = in_shape
            flat_size = int(x / poolsize) * int(y / poolsize)
            self.pool = nn.AvgPool2d(kernel_size=poolsize, stride=poolsize)
        elif len(in_shape) == 3:
            x, y, z = in_shape
            flat_size = int(x / poolsize) * int(y / poolsize) * int(z / poolsize)
            self.pool = nn.AvgPool3d(kernel_size=poolsize, stride=poolsize)
        else:
            raise Exception('Input dimensions higher than 3 not supported (yet)')

        # hypernetwork
        self.tohidden = nn.Sequential(
            Flatten(),
            nn.Linear(flat_size, hidden),
            self.activation
        )

        self.conv1= nn.ConvTranspose1d(in_channels=1, out_channels=self.w_rank+2, kernel_size=2, stride=2)

        self.convs = nn.ModuleList()
        for i in range(deconvs - 1):
            self.convs.append(
                nn.ConvTranspose1d(in_channels=self.w_rank+2, out_channels=self.w_rank+2, kernel_size=2, stride=2))

        self.bias = nn.Sequential(
            nn.Linear(hidden, hyper.prod(out_shape)),
            self.activation
        )

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        insize = input.size()

        downsampled = self.pool(input.unsqueeze(1)).unsqueeze(1)

        hidden = self.tohidden(downsampled)

        res = hidden.unsqueeze(1)

        res = self.conv1(res)

        for i, conv in enumerate(self.convs):
            if i != 0:
                res = self.activation(res)
            res = conv(res)

        res = res.transpose(1,2)
        # res has shape batch_size x k x rank+2

        means, sigmas, values = split_out(res, input.size()[1:], self.out_shape)

        bias = self.bias(hidden)
        # bias = bias.view((-1, ) + self.out_shape)
        # print('down', downsampled)
        # print('params', list(self.tohidden[1].parameters()))
        # print('hidden', hidden)

        return means, sigmas, values, bias

if __name__ == '__main__':

    x = torch.randn(2, 3, 3)
    print(x)
    x = norm(x)
    print(x)

    print(torch.norm(x.view(2, -1), p=2, dim=1))
