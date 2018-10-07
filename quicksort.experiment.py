import gaussian, global_temp
import torch, random, sys
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import util, logging, time, gc
import numpy as np
from scipy.stats import sem
from numpy import std

from argparse import ArgumentParser

import os

from gaussian import HyperLayer

logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Experiment: learn a mapping from a random x, to x sorted.

"""
w = SummaryWriter()

util.DEBUG = False
BUCKET_SIGMA = 0.05

class Split(global_temp.HyperLayer):
    """
    A split matrix moves the elements of the input to either the top or the bottom
    half of a subsection of the output, but keeps the ordering intact otherwise.

    For depth 0, each element is moved to the top or bottom half of the output. For
    depth 1 each element is moved to the top or bottom half of its current half of
    the matrix and so on.

    """
    def __init__(self, size, depth, radditional=1, gadditional=1, learn=0, region=None, sigma_scale=0.1, sigma_floor=0.0):

        init = torch.LongTensor(range(size)).unsqueeze(1).expand(size, 2)

        super().__init__(in_rank=1, out_size=(size,),
                         temp_indices=init, learn_cols=(learn,),
                         gadditional=gadditional, radditional=radditional, region=region,
                         bias_type=gaussian.Bias.NONE, subsample=None, chunk_size=1)

        self.register_buffer('init', init)

        self.size = size
        self.depth = depth
        self.sigma_scale = sigma_scale
        self.sigma_floor = sigma_floor

        indices = torch.arange(size) / size + 1/(2*size)
        self.register_buffer('indices', indices)

        self.sigmas = nn.Parameter(torch.randn(size))
        self.register_buffer('values', torch.ones(size))

    def generate_integer_tuples(self, means, rng=None, use_cuda=False, relative_range=None):


    def hyper(self, input, offset):

        b, s = input.size()
        dv = 'cuda' if self.use_cuda else 'cpu'

        # Split by the given offset vector
        indices = split(self.indices, offset, self.depth)

        # Continue splitting with offsets that maintain the order, so that the indices become equally spaced
        md = int(np.log2(self.size))
        for d in range(self.depth + 1, md):
            id_offset = torch.mod(torch.arange(self.size, device=dv), 2 ** (md - d))
            id_offset = (id_offset + 10e-10) / (2 ** (md - d))
            id_offset = id_offset.round()
            id_offset = id_offset[None, :].expand(b, -1)

            indices = split(indices, id_offset, d)

        d = int(np.ceil(np.log2(s)))
        inf = (0.5 ** d, 1.0 - 0.5 ** d)
        means = util.linmoid(indices, inf_in=inf, up=s-1)

        sigmas = F.softplus(self.sigmas)
        sigmas = sigmas * self.sigma_scale + self.sigma_floor
        sigmas = sigmas[None, :].expand(b, s)

        sigmas = sigmas * (s)

        values = self.values[None, :].expand(b, s)

        return means[:, :, None], sigmas[:, :, None], values

def split(indices, offset, depth):
    b, s = indices.size()

    interval = 2 ** - depth
    mins = torch.floor(indices * 2 ** depth) * interval

    indices = indices - mins
    indices = (indices + offset * interval) / 2.0
    indices = indices + mins

    return indices

class SortLayer(nn.Module):
    """

    """
    def __init__(self, size, learn=0, gadditional=0, radditional=0, region=None, sigma_scale=0.1, sigma_floor=0.0):

        mdepth = int(np.log2(size))

        self.layers = nn.ModuleList()
        for d in range(mdepth):
            self.layers.append(Split(size, d, radditional, gadditional, learn, region, sigma_scale, sigma_floor))

        global_size = 1
        # self.offset = nn.Sequential(
        #     nn.Linear(global_size + 1, 16), nn.ReLU(),
        #     nn.Linear(16, 16), nn.ReLU(),
        #     nn.Linear(16, 1),
        #     nn.Sigmoid())
        self.offset = nn.Sequential(
            util.Lambda(lambda x : x[:, 0] - x[:, 1]),
            util.Lambda(lambda x : x* 50),
            nn.Sigmoid()
        )

        # topivot = nn.Linear(size, global_size, bias=False)
        # topivot.weight.data = topivot.weight.data * 0.00001 + 1.0 / global_size
        # topivot.weight.requires_grad = False
        # topivot = nn.Sequential(
        #     nn.Linear(size, global_size * 2), nn.ReLU(),
        #     nn.Linear(global_size * 2, global_size * 2), nn.ReLU(),
        #     nn.Linear(global_size * 2, global_size)
        # )

    def forward(self, x):

        b, s = x.size()

        for d, split in enumerate(self.layers):

            buckets = x[:, :, None].view(b, 2**d, -1)

            # compute pivots
            pivots = buckets.mean(dim=2, keepdim=True).expand_as(buckets)
            pivots = pivots.view(b, -1)

            # compute offsets by comparing values to pivots
            offset = self.offset(torch.cat([x, pivots], dim=1))

            x = split(x, offset)

        return x

def gen(b, s):
    t = torch.tensor(range(s), dtype=torch.float)[None, :].expand(b, s)/s

    x = torch.zeros(b, s)
    for row in range(b):
        randind = torch.randperm(s)
        x[row, :] = t[row, randind]
    return x

def go(arg):

    MARGIN = 0.1

    torch.manual_seed(arg.seed)

    ndots = arg.iterations // arg.dot_every

    results = np.zeros((arg.reps, ndots))

    print('Starting size {} '.format(arg.size))

    for r in range(arg.reps):
        print('starting {} out of {} repetitions'.format(r, arg.reps))
        util.makedirs('./quicksort/{}'.format( r))
        SHAPE = (arg.size,)

        gaussian.PROPER_SAMPLING = arg.size < 8

        model = SortLayer(arg.size, k=arg.size,
                          gadditional=arg.gadditional, radditional=arg.radditional, region=(arg.chunk,),
                          sigma_scale=arg.sigma_scale, sigma_floor=arg.min_sigma)

        if arg.cuda:
           model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=arg.lr)

        for i in trange(arg.iterations):

            if i > 3000:
                util.DEBUG = True

            x = gen(arg.batch_size, arg.size) # torch.randn((arg.batch_size,) + SHAPE)

            t, idxs = x.sort()

            if arg.cuda:
                x, t = x.cuda(), t.cuda()

            x, t = Variable(x), Variable(t)

            optimizer.zero_grad()

            y = model(x)

            loss = F.mse_loss(y, t) # compute the loss

            loss.backward()

            # print('s', model.sigmas.grad[0])
            # for split in model.splits:
            #     # print('--', split.last.grad)
            #     print('tp', split.topivot.weight.grad)
            # sys.exit()

            optimizer.step()

            w.add_scalar('quicksort/loss/{}/{}'.format(arg.size, r), loss.data.item(), i*arg.batch_size)

            # Compute accuracy estimate
            if i % arg.dot_every == 0:
                with torch.no_grad():

                    correct = 0
                    tot = 0
                    for ii in range(10000//arg.batch_size):
                        x = gen(arg.batch_size, arg.size)
                        t, gold = x.sort()

                        if arg.cuda:
                            x, t = x.cuda(), t.cuda()
                            gold = gold.cuda()

                        x, t = Variable(x), Variable(t)

                        means, sigmas, values = model.hyper(x)
                        m = means.squeeze().round().long()
                        _, m = m.sort()

                        if arg.cuda:
                            m = m.cuda()

                        tot += x.size(0)
                        correct += ((gold != m).sum(dim=1) == 0).sum().item()

                    print('acc', correct/tot)

                    results[r, i//arg.dot_every] = 1.0 - (correct/tot)

                    w.add_scalar('quicksort/accuracy/{}/{}'.format(arg.size, r), correct/tot, i * arg.batch_size)

    np.save('results.{}.np'.format(arg.size), results)
    print('experiments finished')

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    #
    # print(results.shape)
    # print(np.mean(results[:, :], axis=0))
    # print(np.arange(ndots) * arg.dot_every)

    if results.shape[0] > 1:
        ax.errorbar(x=np.arange(ndots) * arg.dot_every, y=np.mean(results[:, :], axis=0),
                        yerr=np.std(results[:, :], axis=0),
                        label='size {0}x{0}, r={1}'.format(arg.size, arg.reps))
    else:
        ax.plot(np.arange(ndots) * arg.dot_every, np.mean(results[:, :], axis=0),
                        label='size {0}x{0}'.format(arg.size))

    ax.legend()

    util.basic(ax)

    ax.spines['bottom'].set_position('zero')
    ax.set_ylim(0.0, 1.0)
#    ax.set_xlim(0.0, 100.0)

    plt.xlabel('iterations')
    plt.ylabel('error')

    plt.savefig('./quicksort/result.png')
    plt.savefig('./quicksort/result.pdf')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Dimensionality of the input.",
                        default=8, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=128, type=int)

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="Number of iterations (in batches).",
                        default=8000, type=int)

    parser.add_argument("-a", "--additional",
                        dest="gadditional",
                        help="Number of additional points sampled globally",
                        default=2, type=int)

    parser.add_argument("-A", "--radditional",
                        dest="radditional",
                        help="Number of additional points sampled regionally",
                        default=2, type=int)

    parser.add_argument("-C", "--chunk",
                        dest="chunk",
                        help="Size of the sampling region",
                        default=4, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-p", "--plot-every",
                        dest="plot_every",
                        help="Plot every x iterations",
                        default=50, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="Random seed.",
                        default=32, type=int)

    parser.add_argument("-d", "--dot-every",
                        dest="dot_every",
                        help="How many iterations per dot in the loss curves.",
                        default=1000, type=int)

    parser.add_argument("-D", "--depth",
                        dest="depth",
                        help="Depth of the sorting network.",
                        default=4, type=int)

    parser.add_argument("-S", "--sigma-scale",
                        dest="sigma_scale",
                        help="Sigma scale.",
                        default=0.1, type=float)

    parser.add_argument("-R", "--repeats",
                        dest="reps",
                        help="Number of repeats.",
                        default=10, type=int)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Sigma floor (minimum sigma value).",
                        default=0.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(options)
