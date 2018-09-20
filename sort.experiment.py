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


class SortLayer(HyperLayer):
    """

    """
    def __init__(self, size, k,  additional=0, sigma_scale=0.1, fix_values=False, sigma_floor=0.0):

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

        activation = nn.Sigmoid()
        hiddenbig = size * 9
        hidden = size * 6

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

        h = size * size * 2
        self.source = nn.Sequential(
            nn.Linear(size, h),
            activation,
            nn.Linear(h, h),
            activation,

            nn.Linear(h, outsize),
        )

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


def go(iterations=30000, batch=4, max_size=16, cuda=False, plot_every=50, lr=0.01, fv=False, seed=0, sigma_scale=0.1,
       reps=10, dot_every=100, sigma_floor=0.0, penalty=0.0):

    MARGIN = 0.1

    torch.manual_seed(seed)

    ndots = iterations//dot_every

    results = np.zeros((max_size-3, reps, ndots))

    if os.path.exists('results.np'):
        results = np.load('results.np')
    else:
        for si, size in enumerate(range(3, max_size)):
            additional = int(np.floor(np.log2(size)) * size)

            print('Starting size {} with {} additional samples '.format(size, additional))
            for r in trange(reps):
                util.makedirs('./sort/{}/{}'.format(size, r))
                SHAPE = (size,)

                gaussian.PROPER_SAMPLING = size < 8

                model = SortLayer(size, k=size, additional=additional, sigma_scale=sigma_scale, fix_values=fv, sigma_floor=sigma_floor)

                if cuda:
                   model.cuda()

                optimizer = optim.Adam(model.parameters(), lr=lr)

                for i in trange(iterations):
                    #
                    # t = torch.tensor(range(size), dtype=torch.float).unsqueeze(0).expand(batch, size)/size
                    #
                    # x = torch.zeros(batch, size)
                    # for row in range(batch):
                    #     randind = torch.randperm(size)
                    #     x[row, :] = t[row, randind]

                    x = torch.randn((batch,) + SHAPE)

                    t, idxs = x.sort()

                    if cuda:
                        x, t = x.cuda(), t.cuda()

                    x, t = Variable(x), Variable(t)

                    optimizer.zero_grad()

                    y = model(x)

                    loss = F.mse_loss(y, t)#, reduce=False).sum(dim=1) # compute the loss
                    #loss = (loss + penalty * model.sigma_loss(x)).sum()

                    loss.backward()        # compute the gradients

                    optimizer.step()

                    w.add_scalar('sort/loss/{}/{}'.format(size, r), loss.data.item(), i*batch)

                    # Compute accuracy estimate
                    if i % dot_every == 0:

                        correct = 0
                        tot = 0
                        for ii in range(10000//batch):
                            x = torch.randn((batch,) + SHAPE)
                            t, idxs = x.sort()

                            if cuda:
                                x, t = x.cuda(), t.cuda()
                            x, t = Variable(x), Variable(t)

                            means, sigmas, values = model.hyper(x)

                            # first example in batch, sort by
                            m = means.round().long()
                            sorted, id = m[:, :, 0].sort()

                            mo = torch.LongTensor(batch, size, 2)
                            for b in range(batch):
                                mo[b, :, :] = m[b, id[b], :]
                            m = mo

                            gold = torch.LongTensor(batch, size, 2)
                            gold[:, :, 0] = torch.tensor(range(size)).unsqueeze(0).expand(batch, size)
                            gold[:, :, 1] = idxs

                            tot += x.size(0)
                            correct += ((gold.view(batch, -1) != m.view(batch, -1)).sum(dim=1) == 0).sum().item()

                            # if ii == 0:
                            #     print( (gold.view(batch, -1) != m.view(batch, -1) ).sum(dim=1) )
                            #
                            #     print(x[0])
                            #     print(gold[0])
                            #     print(means[0])


                        # print('acc', correct/tot)

                        results[si, r, i//dot_every] = 1.0 - (correct/tot)
                        w.add_scalar('sort/accuracy/{}/{}'.format(size, r), correct/tot, i * batch)

                    if i % plot_every == 0:

                        means, sigmas, values = model.hyper(x)

                        plt.figure(figsize=(5, 5))

                        plt.cla()
                        util.plot(means, sigmas, values, shape=(SHAPE[0], SHAPE[0]))
                        plt.xlim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
                        plt.ylim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
                        plt.savefig('./sort/{}/{}/means{:04}.pdf'.format(size, r, i))

        print('experiments finished')

        np.save('results.np', results)

    plt.figure(figsize=(5, 5))
    plt.clf()

    for si, size in enumerate(range(3, max_size)):
        print(sem(results[si, :, :], axis=0))
        plt.errorbar(x=np.arange(ndots) * dot_every, y=np.mean(results[si, :, :], axis=0), yerr=std(results[si, :, :], axis=0), label='{} by {}'.format(size, size))
        plt.legend()

    util.basic()

    plt.savefig('./sort/results.png')
    plt.savefig('./sort/results.pdf')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=128, type=int)

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Max size of the input",
                        default=5, type=int)

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="The number of iterations (ie. the nr of batches).",
                        default=60000, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled",
                        default=10, type=int)

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

    parser.add_argument("-D", "--depth-mult",
                        dest="depth-mult",
                        help="Depth multiplier. The hypernetwork is n layers deep where n = floor(size*depth_mult).",
                        default=1.5, type=float)

    parser.add_argument("-S", "--sigma-scale",
                        dest="sigma_scale",
                        help="Sigma scale.",
                        default=0.1, type=float)

    parser.add_argument("-R", "--repeats",
                        dest="reps",
                        help="Number of repeats.",
                        default=10, type=int)

    parser.add_argument("-P", "--penalty",
                        dest="penalty",
                        help="Sigma penalty.",
                        default=0.0, type=float)

    parser.add_argument("-Q", "--sigma-floor",
                        dest="sigma_floor",
                        help="Sigma floor (minimum sigma value).",
                        default=0.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(batch=options.batch_size, iterations=options.iterations, cuda=options.cuda,
        lr=options.lr, plot_every=options.plot_every, max_size=options.size, fv=options.fix_values,
        seed=options.seed, sigma_scale=options.sigma_scale, reps=options.reps,
       dot_every=options.dot_every, sigma_floor=options.sigma_floor, penalty=options.penalty)
