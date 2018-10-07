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

class Split(nn.Module):

    def __init__(self, size, depth, global_size=None, offset_hidden=None, offset=None, topivot=None):
        super().__init__()

        self.size = size
        self.depth = depth

        if global_size is None:
            global_size = 1 # int(np.log2(size))

        self.global_size = global_size

        if offset_hidden is None:
            offset_hidden = 4 * int(np.log2(size))

        # Computes a global representation of the input sequence (for instance containing the pivots)
        self.topivot = topivot

        # Computes the 'offset' indicating whether an element should be sorted in to the top or bottom half of its current
        # interval
        # self.offset = nn.Sequential(
        #     nn.Linear(1+global_size, offset_hidden),
        #     nn.ReLU(),
        #     nn.Linear(offset_hidden, 1),
        #     nn.Sigmoid()
        # )

        self.offset = offset

        self.offset_scale = Parameter(torch.tensor([50.0]))

        self.register_buffer('buckets0', torch.zeros(1, size))
        self.register_buffer('mins0', torch.zeros(1, 1))
        self.register_buffer('rngs0', torch.ones(1, 1))
        self.register_buffer('prop0', torch.zeros(1, size))
        self.register_buffer('pivots0', torch.zeros(1, size, 1))
        self.register_buffer('pm0', torch.zeros(1, size))
        self.register_buffer('pr0', torch.zeros(1, size))

    def forward(self, i, x, buckets=None):
        """
        :param i: (batched) vector of values in (0, 1), representing indices in the first column
         of the permutation matrix
        :param x: The values to be sorted
        :param depth: depth of the splitting. At d=1 points are moved to the top or the bottom
        half of the (0, 1) rng. At higher depths, points are moved to the top or bottom half
        of the dyadic interval they're in.
        :return:
        """

        b, s = x.size()
        g = self.global_size

        if buckets is None:
            buckets = self.buckets0.expand(b, s)

        # QS:
        # pivots = torch.zeros(b, s)
        # for ba in range(b):
        #     for bu in range(2**(self.depth)):
        #         ids = buckets[ba] == bu
        #         mean_b = x[ba, ids].mean()
        #         pivots[ba, ids] = mean_b

        pivots = self.pivots0.expand(b, s, g)

        for bucket in range(2 ** self.depth):
            # The extent to which each point 'belongs' to the current bucket
            # (the point belongs to the two nearest buckets with extent
            #   proportional to distance)
            weight = 1.0 - torch.abs(buckets - bucket)
            idx = weight > 0.0
            weight[~idx] = 0.0

            # bucketmean = (weight * x).sum(dim=1, keepdim=True).expand(b, s)
            sum = weight.sum(dim=1, keepdim=True).expand(b, g) + 1e-10

            # bucketmean = bucketmean / sum
            piv = self.topivot(x*weight) / sum
            b, g = piv.size()
            # sum = weight.sum(dim=1, keepdim=True).expand(b, g) + 1e-10
            # bucketpiv =  piv / sum

            pivots = pivots + weight[:, :, None].expand(b, s,g) * piv[:, None, :].expand(b, s, g)

        inp = torch.cat([
            x.contiguous().view(b*s, 1),
            pivots.contiguous().view(b*s, g)], dim = 1)
        offset = self.offset(inp).view(b, s)

        if random.random() < 0.005:
            print(offset[0])

        # offset = (x > pivots.squeeze()).float()
        rng = 2 ** - self.depth # size of the current dyadic intervals
        mins = torch.floor(i * 2 ** self.depth) * rng # lower bound of each value's dyadic interval

        # move each point to the top or bottom half of its dyadic interval
        i = i - mins
        i = (i + offset * rng)/2.0
        i = i + mins

        buckets = buckets * 2 + offset

        return i, buckets

class SortLayer(global_temp.HyperLayer):
    """

    """
    def __init__(self, size, k, learn=0, gadditional=0, radditional=0, region=None, sigma_scale=0.1, sigma_floor=0.0):

        # Initial indices: the identity matrix
        init = torch.LongTensor(range(size)).unsqueeze(1).expand(size, 2)

        super().__init__(in_rank=1, out_size=(size,),
                         temp_indices=init, learn_cols=(learn,),
                         gadditional=gadditional, radditional=radditional, region=region,
                         bias_type=gaussian.Bias.NONE, subsample=None, chunk_size=1)

        self.register_buffer('init', init)

        class NoActivation(nn.Module):
            def forward(self, input):
                return input

        self.k = k
        self.size = size
        self.sigma_scale = sigma_scale
        self.sigma_floor = sigma_floor

        global_size = 1

        self.offset =  nn.Sequential(
            nn.Linear(global_size + 1, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1),
            # util.Lambda(lambda x: x.view(-1, size)),
            # util.Lambda(lambda x : x - x.sort(dim=1)[0][:, size//2-1:size//2+1].mean(dim=1, keepdim=True)),
            # util.Lambda(lambda x : x.view(-1, 1)),
            nn.Sigmoid()
        )

        topivot = nn.Linear(size, global_size, bias=False)
        topivot.weight.data = topivot.weight.data * 0.00001 + 1.0/global_size
        topivot.weight.requires_grad = False
        # topivot = nn.Sequential(
        #     nn.Linear(size, global_size * 2), nn.ReLU(),
        #     nn.Linear(global_size * 2, global_size * 2), nn.ReLU(),
        #     nn.Linear(global_size * 2, global_size)
        # )

        self.splits = nn.ModuleList()
        for d in range( int(np.ceil(np.log2(size)))):
            self.splits.append(Split(size=size, depth=d, offset=self.offset, topivot=topivot, global_size=global_size))

        self.sigmas = nn.Parameter(torch.randn(size))
        self.register_buffer('values', torch.ones(size))

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        b, s = input.size()

        # print('i ', input)

        means = self.init[:, 0].float() / s + 1/(2*s)
        # unsqueeze batch
        means = means[None, :].expand(b, s)

        buckets = None
        for split in self.splits:
            means, buckets = split(means, input, buckets)

        sigmas = F.softplus(self.sigmas)
        sigmas = sigmas * self.sigma_scale + self.sigma_floor
        sigmas = sigmas[None, :].expand(b, s)

        values = self.values[None, :].expand(b, s)

        # scale to range
        sigmas = sigmas * (s)

        d = int(np.ceil(np.log2(s)))
        inf = (0.5 ** d, 1.0 - 0.5 ** d)

        # print('m ', means)

        means = util.linmoid(means, inf_in=inf, up=s-1)

        # print('m ', means)
        # print('m ', means.round())
        # print('  ', input.sort()[1])
        # sys.exit()

        return means[:, :, None], sigmas[:, :, None], values

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

                    # print(x[0])
                    # print(t[0])
                    # print(y[0])

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

                        # print('x', x[0])
                        # print('t', gold[0], x[0, gold[0]])
                        # print('m', id[0], x[0, id[0]], means[0, :, 0])
                        # print()
                        # sys.exit()

                        # print(m[0])
                        # print(gold[0])

                        # mo = torch.LongTensor(arg.batch_size, arg.size, 2)
                        # for b in range(arg.batch_size):
                        #     mo[b, :, :] = m[b, id[b], :]
                        # m = mo

                        tot += x.size(0)
                        correct += ((gold != m).sum(dim=1) == 0).sum().item()

                        # if ii == 0:
                        #     print( (gold.view(batch, -1) != m.view(batch, -1) ).sum(dim=1) )
                        #
                        #     print(x[0])
                        #     print(gold[0])
                        #     print(means[0])


                    print('acc', correct/tot)

                    results[r, i//arg.dot_every] = 1.0 - (correct/tot)

                    w.add_scalar('quicksort/accuracy/{}/{}'.format(arg.size, r), correct/tot, i * arg.batch_size)

                if i % arg.plot_every == 0:
                    plt.figure(figsize=(5, 5))

                    means, sigmas, values = model.hyper(x)

                    means, sigmas, values = means.data, sigmas.data, values.data

                    template = model.init.float().unsqueeze(0).expand(means.size(0), means.size(1), 2)
                    template[:, :, model.learn_cols] = means
                    means = template

                    plt.cla()
                    util.plot1dvert(means[0], sigmas[0], values[0], shape=(SHAPE[0], SHAPE[0]))
                    plt.xlim((-MARGIN * (SHAPE[0] - 1), (SHAPE[0] - 1) * (1.0 + MARGIN)))
                    plt.ylim((-MARGIN * (SHAPE[0] - 1), (SHAPE[0] - 1) * (1.0 + MARGIN)))

                    plt.savefig('./quicksort/{}/means{:04}.pdf'.format(r, i))

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
