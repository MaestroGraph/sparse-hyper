from _context import sparse
from sparse import util

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from util import d


class Convolution(nn.Module):
    """
    A "convolution" with a learned sparsity structure (as opposed to a fixed k x k grid)

    A sparse transformation is used to select k pixels from the input. Their channel
    vectors are concatenated and linearly transformed to the desired number of output
    channels (this is done for every output pixel).

    The pattern of input pixels, relative to the current output pixels is determined
    adaptively.
    """
    def __init__(self, in_size, out_size, k, gadditional, radditional, region, min_sigma=0.05, sigma_scale=1.0, mmult=1.0):
        """
        :param k: Number of connections to the input in total
        :param gadditional:
        :param radditional:
        :param region:
        """

        super().__init__()

        self.in_size, self.out_size = in_size, out_size
        self.min_sigma, self.sigma_scale = min_sigma, sigma_scale
        self.mmult = mmult

        self.k = k

        cin, hin, win = in_size
        cout, hout, wout = out_size

        self.unify = nn.Linear(k*cin, cout)

        # network that generates the coordinates and sigmas
        hidden = cin * 4
        self.toparams = nn.Sequential(
            nn.Linear(cin + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, k * 3) # two means, one sigma
        )

        self.register_buffer('mvalues', torch.ones((k,)))
        self.register_buffer(util.coordinates((hin, win)), 'coords')

        assert self.coords.size() == (2, hin, win)

    def hyper(self, x):

        assert x.size() == self.in_size
        b, c, h, w = x.size()
        k = self.k

        # the coordinates of the current pixels in parameters space
        # - the index tuples are described relative to these
        mids = self.coords[None, :, :, :].expand(b, 2, h, w)
        mids = util.inv(mids.transpose(1, 2, 0), mx=torch.tensor((h, w), device=d(x), dtype=torch.long))
        mids = mids.transpose(2, 0, 1)

        # add coords to channels
        coords = self.coords[None, :, :, :].expand(b, 2, h, w)
        x = torch.cat([x, coords], dim=1)

        x = x.permute(0, 2, 3, 1)

        params = self.toparams(x)
        assert params.size() == (b, h, w, k * 3) # k index tuples per output pixel

        means  = params[:, :, :, :k*2].view(b, h, w, k, 2)
        sigmas = params[:, :, :, k*2:].view(b, h, w, k)
        values = self.mvalues[None, None, None, :].expand(b, h, w, k)

        means = mids + self.mmult * means

        s = (h, w)
        means, sigmas = sparse.transform_means(means, s), \
                        sparse.transform_sigmas(sigmas, s, min_sigma=self.min_sigma) * self.sigma_scale

        return means, sigmas, values

    def forward(self, x):

        assert x.size() == self.in_size
        b, c, h, w = x.size()
        s = (h, w)

        means, sigmas, mvalues = self.hyper(x)

        # sample integer indices and values
        indices = sparse.ngenerate(means, self.gadditional, self.radditional, rng=s, relative_range=(self.region, self.region), cuda=x.is_cuda)

        vs = (4 + self.radditional + self.gadditional)
        assert indices.size() == (b, h, w, k*vs, 2), f'{indices.size()}, {(b, h, w, k*vs, 2)}'

        indices = indices.view(b, h, w, k, vs, 2)

        indfl = indices.float()


def go(arg):
    pass

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=1_000_000, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples per output pixel.",
                        default=32, type=int)

    parser.add_argument("-a", "--gadditional",
                        dest="gadditional",
                        help="Number of additional points sampled globally",
                        default=8, type=int)

    parser.add_argument("-A", "--radditional",
                        dest="radditional",
                        help="Number of additional points sampled locally",
                        default=8, type=int)

    parser.add_argument("-R", "--region",
                        dest="region",
                        help="Size of the (square) region to use for local sampling.",
                        default=8, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-D", "--data", dest="data",
                        help="Dataset (cifar10)",
                        default='cifar10')

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-S", "--subsample",
                        dest="subsample",
                        help="Sample a subset of the indices to estimate gradients for",
                        default=None, type=float)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Minimum value of sigma.",
                        default=0.01, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many batches between tests.",
                        default=1000, type=int)

    parser.add_argument("--plot-every",
                        dest="plot_every",
                        help="How many batches between plotting the sparse indices.",
                        default=100, type=int)

    parser.add_argument("--test-subset",
                        dest="test_subset",
                        help="A subset for the validation tests.",
                        default=100000, type=int)

    parser.add_argument("--test-batchsize",
                        dest="test_batchsize",
                        help="Batch size for computing the validation loss.",
                        default=1024, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
