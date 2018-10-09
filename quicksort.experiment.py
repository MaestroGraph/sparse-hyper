import sort
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

def gen(b, s):
    t = torch.tensor(range(s), dtype=torch.float)[None, :].expand(b, s)/s

    x = torch.zeros(b, s)
    for row in range(b):
        randind = torch.randperm(s)
        x[row, :] = t[row, randind]

    return x[:, :, None]

def go(arg):

    MARGIN = 0.1

    torch.manual_seed(arg.seed)

    ndots = arg.iterations // arg.dot_every

    results = np.zeros((arg.reps, ndots))

    print('Starting size {} '.format(arg.size))

    for r in range(arg.reps):
        print('starting {} out of {} repetitions'.format(r, arg.reps))
        util.makedirs('./quicksort/{}'.format( r))

        model = sort.SortLayer(arg.size,
                          additional=arg.additional, sigma_scale=arg.sigma_scale, sigma_floor=arg.min_sigma)

        if arg.cuda:
           model.cuda()

        tokeys = nn.Linear(arg.size, arg.size, bias=False)

        optimizer = optim.Adam(list(model.parameters()) + list(tokeys.parameters()), lr=arg.lr)

        for i in trange(arg.iterations):

            if i > 3000:
                util.DEBUG = True

            x = gen(arg.batch_size, arg.size) # torch.randn((arg.batch_size,) + SHAPE)

            # keys.requires_grad = True

            t, idxs = x.sort(dim=1)

            if arg.cuda:
                x, t = x.cuda(), t.cuda()

            x, t = Variable(x), Variable(t)

            optimizer.zero_grad()

            keys = tokeys(x.squeeze())
            y, _ = model(x, keys=keys)

            loss = F.mse_loss(y, t) # compute the loss

            loss.backward()

            # print(tokeys.weight.grad)
            # print(keys.grad)
            # # print(model.last.grad)
            # print('x', x[0].grad)
            # print(model.certainty.grad)
            #
            # sys.exit()

            # print('s', model.sigmas.grad[0])
            # for split in model.splits:
            #     # print('--', split.last.grad)
            #     print('tp', split.topivot.weight.grad)
            # sys.exit()

            optimizer.step()

            w.add_scalar('quicksort/loss/{}/{}'.format(arg.size, r), loss.data.item(), i*arg.batch_size)

            if i % arg.dot_every == 0:
                print(tokeys.weight)

            # Compute accuracy estimate
            if i % arg.dot_every == 0 and False:
                with torch.no_grad():

                    losses = []
                    for ii in range(10000//arg.batch_size):
                        x = gen(arg.batch_size, arg.size)
                        t, _ = x.sort(dim=1)

                        if arg.cuda:
                            x, t = x.cuda(), t.cuda()

                        x, t = Variable(x), Variable(t)

                        # keys = x.squeeze()
                        keys = tokeys(x.squeeze())

                        y, _ = model(x, keys=keys)

                        loss = F.mse_loss(y, t)

                        losses.append(loss.item())

                    print('loss', np.mean(losses))

                    results[r, i//arg.dot_every] = np.mean(losses)

                    w.add_scalar('quicksort/accuracy/{}/{}'.format(arg.size, r), np.mean(losses), i * arg.batch_size)

    np.save('results.{}.np'.format(arg.size), results)
    print('experiments finished')

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

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
                        dest="additional",
                        help="Number of additional points sampled globally",
                        default=2, type=int)

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
