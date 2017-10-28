from time import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.axes import Axes
import os, errno, random

import torch
from torch import nn

import subprocess

import numpy as np

tics = []

def tic():
    tics.append(time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time()-tics.pop()

def plot(means, sigmas, values, axes=None):

    b, n, d = means.size()

    means = means.data[0, :,:].cpu().numpy()
    sigmas = sigmas.data[0, :].cpu().numpy()
    values = nn.functional.tanh(values).data[0, :].cpu().numpy()

    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = mpl.cm.RdYlBu
    map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if axes is None:
        axes = plt.gca()

    colors = []
    for i in range(n):
        color = map.to_rgba(values[i])
        alpha = (sigmas[i]+1.0)**-2
        axes.add_patch(Circle((means[i, :]), radius=sigmas[i], color=color, alpha=alpha, linewidth=0))
        colors.append(color)

    axes.scatter(means[:, 0], means[:, 1], c=colors, zorder=100, linewidth=1, edgecolor='k')

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

class SparseMult(torch.autograd.Function):

    def __init__(self, use_cuda=False):
        super().__init__()
        self.use_cuda = use_cuda

        self.FT =  torch.cuda.sparse.FloatTensor if self.use_cuda else torch.sparse.FloatTensor

    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    def forward(self, indices, values, size, vector):

        matrix = self.FT(indices, values, torch.Size(size))

        self.save_for_backward(indices, values, size, vector)
        res = torch.mm(matrix, vector.unsqueeze(1))
        return res

    def backward(self, grad_output):

        indices, values, size, vector = self.saved_tensors
        matrix = self.FT(indices, values, torch.Size(size))

        i_ixs = indices[0,:]
        j_ixs = indices[1,:]
        output_select = grad_output.view(-1)[i_ixs]
        vector_select = vector.view(-1)[j_ixs]

        grad_values = output_select *  vector_select

        grad_vector = torch.mm(matrix.t(), grad_output).t() \
            if self.needs_input_grad[1] else None

        return None, grad_values, None, grad_vector

def nvidia_smi():
    command = 'nvidia-smi'
    return subprocess.check_output(command, shell=True)