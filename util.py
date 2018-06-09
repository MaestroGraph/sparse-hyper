from time import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, Ellipse, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.axes import Axes
import os, errno, random, time, string

import torch
from torch import nn
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch.utils.data import sampler

from collections import OrderedDict

import subprocess

import numpy as np

import math

tics = []

def tic():
    tics.append(time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time()-tics.pop()

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    axes.spines["right"].set_visible(False)
    axes.spines["top"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.spines["left"].set_visible(False)

    # axes.get_xaxis().set_tick_params(which='both', top='off', bottom='off', labelbottom='off')
    # axes.get_yaxis().set_tick_params(which='both', left='off', right='off')


def plot(means, sigmas, values, shape=None, axes=None, flip_y=None, alpha_global=1.0):
    """

    :param means:
    :param sigmas:
    :param values:
    :param shape:
    :param axes:
    :param flip_y: If not None, interpreted as the max y value. y values in the scatterplot are
            flipped so that the max is equal to zero and vice versa.
    :return:
    """

    b, n, d = means.size()

    means = means.data[0, :,:].cpu().numpy()
    sigmas = sigmas.data[0, :].cpu().numpy()
    values = nn.functional.tanh(values).data[0, :].cpu().numpy()

    if flip_y is not None:
        means[:, 0] = flip_y - means[:, 0]

    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = mpl.cm.RdYlBu
    map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if axes is None:
        axes = plt.gca()

    colors = []
    for i in range(n):
        color = map.to_rgba(values[i])
        alpha = max(0.05, ((sigmas[i, 0] * sigmas[i, 0])+1.0)**-2) * alpha_global
        axes.add_patch(Ellipse((means[i, 1], means[i, 0]), width=sigmas[i,1], height=sigmas[i,0], color=color, alpha=alpha, linewidth=0))
        colors.append(color)

    axes.scatter(means[:, 1], means[:, 0],c=colors, zorder=100, linewidth=1, edgecolor='k', alpha=alpha_global)

    if shape is not None:

        m = max(shape)
        step = 1 if m < 100 else m//25

        # gray points for the integer index tuples
        x, y = np.mgrid[0:shape[0]:step, 0:shape[1]:step]
        axes.scatter(x.ravel(),  y.ravel(), c='k', s=5, marker='D', zorder=-100, linewidth=0, alpha=0.1* alpha_global)

    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)


def plot1d(means, sigmas, values, shape=None, axes=None):

    h = 0.1

    n, d = means.size()

    means = means.cpu().numpy()
    sigmas = sigmas.cpu().numpy()
    values = nn.functional.tanh(values).data.cpu().numpy()

    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = mpl.cm.RdYlBu
    map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if axes is None:
        axes = plt.gca()

    colors = []
    for i in range(n):
        color = map.to_rgba(values[i])
        alpha = 0.7 # max(0.05, (sigmas[i, 0]+1.0)**-1)
        axes.add_patch(Rectangle(xy=(means[i, 1]  - sigmas[i, 0]*0.5, means[i, 0] - h*0.5), width=sigmas[i,0] , height=h, color=color, alpha=alpha, linewidth=0))
        colors.append(color)

    axes.scatter(means[:, 1], means[:, 0], c=colors, zorder=100, linewidth=1, edgecolor='k')

    if shape is not None:

        m = max(shape)
        step = 1 if m < 100 else m//25

        # gray points for the integer index tuples
        x, y = np.mgrid[0:shape[0]:step, 0:shape[1]:step]
        axes.scatter(x.ravel(),  y.ravel(), c='k', s=5, marker='D', zorder=-100, linewidth=0, alpha=0.1)

    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

def norm(x):
    """
    Normalize a tensor to a tensor with unit norm (treating first dim as batch dim)

    :param x:
    :return:
    """
    b = x.size()[0]

    n = torch.norm(x.view(b, -1), p=2, dim=1)
    while len(n.size()) < len(x.size()):
        n = n.unsqueeze(1)

    n.expand_as(x)

    return x/n

def makedirs(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def sample(collection, k, required):
    """
    Sample, without replacement, k elements from 'collection', ensuring that 'required' are always contained in the
    sample (but never twice).

    currently only works if collection and required contain only unique elements
    :param k:
    :param collection:
    :param required:
    :return:
    """
    if(k + len(required) > len(collection)):
        # use rejection sampling
        sample = list(collection)
        while len(sample) > k:
            ri = random.choice(range(len(sample)))

            if sample[ri] not in required:
                del(sample[ri])

        return sample
    else:
        required = set(required)
        sample0 = set(random.sample(collection, k + len(required)))
        sample = list(sample0 - required)

        while len(sample) > k - len(required):
            ri = random.choice(range(len(sample)))
            del(sample[ri])

        sample.extend(required)

        return sample

if __name__ == '__main__':

    print('.')
    print(sample(range(6), 5, [0, 1, 2]))
    print('.')
    print(sample(range(100), 6, [0, 1, 2]))
    print(sample(range(100), 6, [0, 1, 2]))
    print(sample(range(100), 6, [0, 1, 2]))
    print('.')

def sparsemult(use_cuda):
    return SparseMultGPU.apply if use_cuda else SparseMultCPU.apply

class SparseMultCPU(torch.autograd.Function):

    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, vector):

        # print(type(size), size, list(size), intlist(size))

        matrix = torch.sparse.FloatTensor(indices, values, torch.Size(intlist(size)))

        ctx.indices, ctx.matrix, ctx.vector = indices, matrix, vector

        return torch.mm(matrix, vector.unsqueeze(1))

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data

        # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices

        i_ixs = ctx.indices[0,:]
        j_ixs = ctx.indices[1,:]
        output_select = grad_output.view(-1)[i_ixs]
        vector_select = ctx.vector.view(-1)[j_ixs]

        grad_values = output_select *  vector_select

        grad_vector = torch.mm(ctx.matrix.t(), grad_output).t()
        return None, Variable(grad_values), None, Variable(grad_vector)

class SparseMultGPU(torch.autograd.Function):

    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, vector):

        # print(type(size), size, list(size), intlist(size))

        matrix = torch.cuda.sparse.FloatTensor(indices, values, torch.Size(intlist(size)))

        ctx.indices, ctx.matrix, ctx.vector = indices, matrix, vector

        return torch.mm(matrix, vector.unsqueeze(1))

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data

        # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices

        i_ixs = ctx.indices[0,:]
        j_ixs = ctx.indices[1,:]
        output_select = grad_output.view(-1)[i_ixs]
        vector_select = ctx.vector.view(-1)[j_ixs]

        grad_values = output_select *  vector_select

        grad_vector = torch.mm(ctx.matrix.t(), grad_output).t()
        return None, Variable(grad_values), None, Variable(grad_vector)

def nvidia_smi():
    command = 'nvidia-smi'
    return subprocess.check_output(command, shell=True)

def orth_loss(batch_size, x_size, model, use_cuda):
    """

    :param batch_size:
    :param x_size:
    :param model:
    :param use_cuda:
    :return:
    """

    x_size = (batch_size,) + x_size

    x1o, x2o = torch.randn(x_size), torch.randn(x_size)

    # normalize to unit tensors
    x1o, x2o = norm(x1o), norm(x2o)

    if use_cuda:
        x1o, x2o = x1o.cuda(), x2o.cuda()
    x1o, x2o = Variable(x1o), Variable(x2o)

    y1 = model(x1o)
    y2 = model(x2o)

    x1 = x1o.view(batch_size, 1, -1)
    x2 = x2o.view(batch_size, 1, -1)
    y1 = y1.view(batch_size, 1, -1)
    y2 = y2.view(batch_size, 1, -1)

    print('x1 v y1', x1[0, :], y1[0, ])

    xnorm = torch.bmm(x1, x2.transpose(1, 2))
    ynorm = torch.bmm(y1, y2.transpose(1, 2))

    loss = torch.sum(torch.pow((xnorm - ynorm), 2)) / batch_size

    return loss, x1o, x2o

def bmultinomial(mat, num_samples=1, replacement=False):
    """
    Take multinomial samples from a batch of matrices with multinomial parameters on the
    rows

    :param mat:
    :param num_samples:
    :param replacement:
    :return:
    """

    batches, rows, columns = mat.size()

    mat = mat.view(1, -1, columns).squeeze(0)

    sample = torch.multinomial(mat, num_samples, replacement)

    return sample.view(batches, rows, num_samples), sample

def bsoftmax(input):

    b, r, c = input.size()
    input = input.view(1, -1, c)
    input = nn.functional.softmax(input.squeeze(0)).unsqueeze(0)

    return input.view(b, r, c)

def contains_nan(tensor):
    return (tensor != tensor).sum() > 0

if __name__ == '__main__':


    i = torch.LongTensor([[0, 16, 1],
                          [2, 0, 2]])
    v = torch.FloatTensor([1, 1, 1])

    matrix = torch.sparse.FloatTensor(i, v, torch.Size((16, 16)))

def od(lst):
    od = OrderedDict()
    for i, elem in enumerate(lst):
        od[str(i)] = elem

    return od

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class Debug(nn.Module):
    def __init__(self, lambd):
        super(Debug, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        self.lambd(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def flatten(input):
    return input.view(input.size(0), -1)

class NoActivation(nn.Module):
    def forward(self, input):
        return input

def prod(tuple):
    result = 1

    for v in tuple:
        result *= v

    return result

def add_noise(input, std=0.1):
    """
    In-place
    :param input:
    :param std:
    :return:
    """

    noise = torch.cuda.FloatTensor(input.size()) if input.is_cuda else FloatTensor(input.size())
    noise.normal_(std=std)

    return input + noise

def corrupt_(input, prop=0.3):
    """
    Sets a random proportion of the input to zero
    :param input:
    :param prop:
    :return:
    """

    t0 = time.time()
    FT = torch.cuda.FloatTensor if input.is_cuda else torch.FloatTensor
    mask = FT(input.size())
    mask.uniform_()

    mask.sub_(prop).ceil_()

    input.mul_(mask)


def rstring(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

def count_params(model):
    sum = 0
    for tensor in model.parameters():
        sum += prod(tensor.size())

    return sum

def logit(x):
    if type(x) == float:
        return math.log(x / (1 - x))
    return torch.log(x/ (1 - x))


def inv(i):
    sc = (i/27) * 0.9999 + 0.00005
    return logit(sc)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset, using a fixed permutation

    initial source: https://github.com/pytorch/vision/issues/168

    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self,  start, num, total, seed = 0):
        self.start = start
        self.num = num

        self.random = random.Random(seed)

        self.l = list(range(total))
        self.random.shuffle(self.l)

    def __iter__(self):

        return iter(self.l[self.start : self.start + self.num])

    def __len__(self):
        return self.num

def bmult(width, height, num_indices, batchsize, use_cuda):
    """
    ?

    :param width:
    :param height:
    :param num_indices:
    :param batchsize:
    :param use_cuda:
    :return:
    """

    bmult = torch.cuda.LongTensor([height, width]) if use_cuda else LongTensor([height, width])
    m = torch.cuda.LongTensor(range(batchsize)) if use_cuda else LongTensor(range(batchsize))

    bmult = bmult.unsqueeze(0).unsqueeze(0)
    m     = m.unsqueeze(1).unsqueeze(1)

    bmult = bmult.expand(batchsize, num_indices, 2)
    m     = m.expand(batchsize, num_indices, 2)

    return m * bmult

def intlist(tensor):
    """
    A slow and stupid way to turn a tensor into an iterable over ints
    :param tensor:
    :return:
    """

    tensor = tensor.squeeze()

    assert len(tensor.size()) == 1

    s = tensor.size()[0]

    l = [None] * s
    for i in range(s):
        l[i] = int(tensor[i])

    return l



