import gaussian
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

def penalty(means):
    b, k, r = means.size()
    sorted, _ = means.sort(dim=1)
    diffs = sorted[:, :-1, :] - sorted[:, :1, :]

    penalty = 1.0 - diffs[diffs < 1.0]

    # print(sorted.size(), sorted)
    # print('d', diffs)
    # print(diffs[diffs < 1.0])
    # print(penalty.size(), penalty)

    # sys.exit()

    return penalty.mean()

class SortLayer(HyperLayer):
    """

    """
    def __init__(self, size, k,  additional=0, sigma_scale=0.1, fix_values=False, sigma_floor=0.0, depth=4):

        if size < 8:
            gaussian.PROPER_SAMPLING = True

        super().__init__(in_rank=1, out_shape=(size,), additional=additional, bias_type=gaussian.Bias.NONE, subsample=None)

        class NoActivation(nn.Module):
            def forward(self, input):
                return input

        self.k = k
        self.size = size
        self.sigma_scale = sigma_scale
        self.sigma_floor = sigma_floor
        self.fix_values = fix_values

        outsize = 4 * k
        h = size * size * 2
        activation = nn.ReLU()

        # self.source = nn.Sequential(
        #     nn.Linear(size, hiddenbig),
        #     activation,
        #     nn.Linear(hiddenbig, hiddenbig),
        #     activation,
        #     nn.Linear(hiddenbig, hiddenbig),
        #     activation,
        #     nn.Linear(hiddenbig, hidden),
        #     activation,
        #     nn.Linear(hidden, hidden),
        #     activation,
        #     nn.Linear(hidden, hidden),
        #     activation,
        #     nn.Linear(hidden, outsize),
        # )

        layers = []
        layers.append(nn.Linear(size, h))
        layers.append(activation)

        for _ in range(depth):
            layers.append(nn.Linear(h, h, bias=True))
            # layers.append(nn.BatchNorm1d(HIDDENBIG))
            layers.append(activation)

        layers.append(nn.Linear(h, outsize))

        self.source = nn.Sequential(*layers)

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """
        b, s = input.size()

        res = self.source(input).unsqueeze(2).view(b, self.k, 4)

        self.sigloss = res[:, :, 2].log().sum(dim=1)

        means, sigmas, values = self.split_out(res, input.size()[1:], self.out_size)

        sigmas = sigmas * self.sigma_scale + self.sigma_floor


        if self.fix_values:
            values = values * 0.0 + 1.0

        return means, sigmas, values

    # def sigma_loss(self):
    #     return self.sigloss

    def sigma_loss(self, input):
        b, s = input.size()

        res = self.source(input).unsqueeze(2).view(b, self.k, 4)
        means, sigmas, values = self.split_out(res, input.size()[1:], self.out_size)

        return - torch.log(sigmas.sum(dim=2).sum(dim=1) / (self.k * s))


def go(arg):

    MARGIN = 0.1

    torch.manual_seed(arg.seed)

    ndots = arg.iterations // arg.dot_every

    additional = int(np.floor(np.log2(arg.size)) * arg.size) if arg.additional is None else arg.additional

    results = np.zeros((arg.reps, ndots))

    print('Starting size {} with {} additional samples '.format(arg.size, additional))

    for r in range(arg.reps):
        print('starting {} out of {} repetitions'.format(r, arg.reps))
        util.makedirs('./sort/{}'.format( r))
        SHAPE = (arg.size,)

        gaussian.PROPER_SAMPLING = arg.size < 8

        model = SortLayer(arg.size, k=arg.size, additional=additional, sigma_scale=arg.sigma_scale, fix_values=arg.fix_values,
                          sigma_floor=arg.min_sigma, depth=arg.depth)

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

            loss = F.mse_loss(y, t)#, reduce=False).sum(dim=1) # compute the loss
            #loss = (loss + penalty * model.sigma_loss(x)).sum()

            if arg.penalty is not None:
                means, _, _ = model.hyper(x) # TODO double computation
                loss = loss + arg.penalty * penalty(means)

            loss.backward()        # compute the gradients

            optimizer.step()

            w.add_scalar('sort/loss/{}/{}'.format(arg.size, r), loss.data.item(), i*arg.batch_size)

            # Compute accuracy estimate
            if i % arg.dot_every == 0:

                correct = 0
                tot = 0
                for ii in range(10000//arg.batch_size):
                    x = torch.randn((arg.batch_size,) + SHAPE)
                    t, idxs = x.sort()

                    if arg.cuda:
                        x, t = x.cuda(), t.cuda()
                    x, t = Variable(x), Variable(t)

                    means, sigmas, values = model.hyper(x)

                    # first example in batch, sort by
                    m = means.round().long()
                    sorted, id = m[:, :, 0].sort()

                    mo = torch.LongTensor(arg.batch_size, arg.size, 2)
                    for b in range(arg.batch_size):
                        mo[b, :, :] = m[b, id[b], :]
                    m = mo

                    gold = torch.LongTensor(arg.batch_size, arg.size, 2)
                    gold[:, :, 0] = torch.tensor(range(arg.size)).unsqueeze(0).expand(arg.batch_size, arg.size)
                    gold[:, :, 1] = idxs

                    tot += x.size(0)
                    correct += ((gold.view(arg.batch_size, -1) != m.view(arg.batch_size, -1)).sum(dim=1) == 0).sum().item()

                    # if ii == 0:
                    #     print( (gold.view(batch, -1) != m.view(batch, -1) ).sum(dim=1) )
                    #
                    #     print(x[0])
                    #     print(gold[0])
                    #     print(means[0])


                print('acc', correct/tot)

                results[r, i//arg.dot_every] = 1.0 - (correct/tot)

                w.add_scalar('sort/accuracy/{}/{}'.format(arg.size, r), correct/tot, i * arg.batch_size)

            if i % arg.plot_every == 0:

                means, sigmas, values = model.hyper(x)

                plt.figure(figsize=(5, 5))

                plt.cla()
                util.plot(means, sigmas, values, shape=(SHAPE[0], SHAPE[0]))
                plt.xlim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
                plt.ylim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
                plt.savefig('./sort/{}/means{:04}.pdf'.format(r, i))

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
                        label='size {0}x{0}, a={1}, r={2}'.format(arg.size, additional, arg.reps))
    else:
        ax.plot(np.arange(ndots) * arg.dot_every, np.mean(results[:, :], axis=0),
                        label='size {0}x{0}, a={1}'.format(arg.size, additional))

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
                        dest="additional",
                        help="Number of additional points sampled",
                        default=None, type=int)

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

    parser.add_argument("-F", "--fix_values", dest="fix_values",
                        help="Whether to fix the values to 1.",
                        action="store_true")

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

    parser.add_argument("-P", "--penalty",
                        dest="penalty",
                        help="Penalty.",
                        default=None, type=float)


    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(options)
