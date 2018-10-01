import gaussian, gaussian_temp
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

class Split(nn.Module):

    def __init__(self, size, depth, global_size=None, offset_hidden=None):
        super().__init__()

        self.size = size
        self.depth = depth

        if global_size is None:
            global_size = 4 * int(np.log2(size))

        if offset_hidden is None:
            offset_hidden = 2 * global_size

        # Computes a global representation of the input sequence (for instance containing the pivots)
        self.globalrep = nn.Sequential(
            nn.Linear(size * 2, 2* global_size),
            nn.ReLU(),
            nn.Linear(2* global_size, global_size)
        )

        # Computes the 'offset' indicating whether an element should be sorted in to the top or bottom half of its current
        # interval
        self.offset = nn.Sequential(
            nn.Linear(global_size+1, offset_hidden*2),
            nn.ReLU(),
            nn.Linear(offset_hidden*2, offset_hidden),
            nn.ReLU(),
            nn.Linear(offset_hidden, 1),
            #util.Lambda(lambda x : x * 0.0001),
            nn.Sigmoid()
        )

        self.register_buffer('buckets0', torch.zeros(1, size))

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

        # SQ: These snippets show code that computes quicksort explicitly
        b, s = x.size()

        if buckets is None:
            buckets = self.buckets0.expand(b, s)

        rng = 2 ** - self.depth
        mins = torch.floor(i * 2 ** self.depth) / 2 ** self.depth
        # - lower bound of each value's dyadic interval

        # compute 'offset': Vector of values in (0, 1) indicating whether points should be moved to the top or bottom half of the
        # rng (expected to converge to values close to 0 or 1)
        gr = self.globalrep( torch.cat((x, buckets), dim=1) )
        b, g = gr.size()

        # expand for each element in each instance
        gr = gr.unsqueeze(1).expand(b, s, g)

        # fold the size dimension into the batch dimension and concatenate
        inp = torch.cat([x.contiguous().view(-1, 1), gr.contiguous().view(-1, g)], dim = 1)
        offset = self.offset(inp).view(b, s)

        # QS: if pivots is None:
        #     pivots = x.median(dim=1, keepdim=True)[0].expand(b, s)

        # QS: offset = (x > pivots).float()

        i = i - mins
        i = (i + offset * rng)/2.0
        i = i + mins

        buckets = buckets * 2 + offset

        # QS: stores the mean of the bucket each element is in
        # pivots = torch.zeros(b, s, self.f)
        # for ba in range(b):
        #     for bu in range(2**(self.depth+1)):
        #         ids = buckets[ba] == bu
        #         mean_b = x[ba, ids].median()[0]
        #         pivots[ba, ids] = mean_b

        return i, buckets #, pivots

class SortLayer(gaussian_temp.HyperLayer):
    """

    """
    def __init__(self, size, k, learn=0, gadditional=0, radditional=0, region=None, sigma_scale=0.1, sigma_floor=0.0):

        # Initial indices: the identity matrix
        init = torch.LongTensor(range(size)).unsqueeze(1).expand(size, 2)

        super().__init__(in_rank=1, out_size=(size,),
                         temp_indices=init, learn_cols=(learn,),
                         gadditional=gadditional, radditional=radditional, region=region,
                         bias_type=gaussian.Bias.NONE, subsample=None)

        self.register_buffer('init', init)

        class NoActivation(nn.Module):
            def forward(self, input):
                return input

        self.k = k
        self.size = size
        self.sigma_scale = sigma_scale
        self.sigma_floor = sigma_floor

        self.splits = nn.ModuleList()
        for d in range(int(np.ceil(np.log2(size)))):
            self.splits.append(Split(size=size, depth=d))

        # print('depth ', len(self.splits))

        self.sigmas = nn.Parameter(torch.randn(size))
        self.register_buffer('values', torch.ones(size))

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """
        b, s = input.size()

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
        means = util.linmoid(means, inf_in=inf, up=s-1)

        return means[:, :, None], sigmas[:, :, None], values

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
            #
            # t = torch.tensor(range(size), dtype=torch.float).unsqueeze(0).expand(batch, size)/size
            #
            # x = torch.zeros(batch, size)
            # for row in range(batch):
            #     randind = torch.randperm(size)
            #     x[row, :] = t[row, randind]

            x = torch.randn((arg.batch_size,) + SHAPE)

            t, idxs = x.sort()

            if arg.cuda:
                x, t = x.cuda(), t.cuda()

            x, t = Variable(x), Variable(t)

            optimizer.zero_grad()

            y = model(x)

            loss = F.mse_loss(y, t) # compute the loss

            loss.backward()
            optimizer.step()

            w.add_scalar('quicksort/loss/{}/{}'.format(arg.size, r), loss.data.item(), i*arg.batch_size)

            # Compute accuracy estimate
            if i % arg.dot_every == 0:

                # print(x[0])
                # print(t[0])
                # print(y[0])

                correct = 0
                tot = 0
                for ii in range(10000//arg.batch_size):
                    x = torch.randn((arg.batch_size,) + SHAPE)
                    t, gold = x.sort()

                    if arg.cuda:
                        x, t = x.cuda(), t.cuda()

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

    plt.savefig('./sort/result.png')
    plt.savefig('./sort/result.pdf')

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
