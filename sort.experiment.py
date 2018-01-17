import hyper, gaussian
import torch, random, sys
from torch.autograd import Variable
from torch.nn import Parameter
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import util, logging, time, gc
import numpy as np

from argparse import ArgumentParser

import psutil, os

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
    def __init__(self, size, k, additional=0, sigma_scale=0.1, fix_values=False):

        super().__init__(in_rank=1, out_shape=(size,), additional=additional, bias_type=gaussian.Bias.NONE, subsample=None)

        class NoActivation(nn.Module):
            def forward(self, input):
                return input


        self.k = k
        self.size = size
        self.sigma_scale = sigma_scale
        self.fix_values = fix_values

        outsize = 4 * k

        activation = nn.ReLU()
        self.source = nn.Sequential(
            nn.Linear(size, size * 3),
            activation,
            nn.Linear(size * 3, size * 3),
            activation,
            nn.Linear(size * 3, outsize),
        )

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """
        b, s = input.size()

        res = self.source(input).unsqueeze(2).view(b, self.k, 4)

        means, sigmas, values = self.split_out(res, input.size()[1:], self.out_shape)
        sigmas = sigmas * self.sigma_scale

        if self.fix_values:
            values = values * 0.0 + 1.0

        return means, sigmas, values

    def sigma_loss(self, input):
        b, s = input.size()

        res = self.source(input).unsqueeze(2).view(b, self.k, 4)
        means, sigmas, values = self.split_out(res, input.size()[1:], self.out_shape)

        return - torch.log(sigmas.sum()/self.k)


def go(iterations=30000, additional=64, batch=4, size=32, cuda=False, plot_every=50, lr=0.01, fv=False, seed=0, sigma_scale=0.1):

    SHAPE = (size,)
    MARGIN = 0.1

    torch.manual_seed(seed)

    nzs = hyper.prod(SHAPE)

    plt.figure(figsize=(5,5))
    util.makedirs('./sort/')

    params = None

    gaussian.PROPER_SAMPLING = True
    model = SortLayer(size, k=size, additional=additional, sigma_scale=sigma_scale, fix_values=fv)

    if cuda:
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for i in trange(iterations):

        x = torch.randn((batch,) + SHAPE)

        t = x.sort()[0]

        if cuda:
            x, t = x.cuda(), t.cuda()

        x, t = Variable(x), Variable(t)

        optimizer.zero_grad()

        y = model(x)

        loss = criterion(y, t) + 0.005 * model.sigma_loss(x) # compute the loss

        t0 = time.time()
        loss.backward()        # compute the gradients

        optimizer.step()

        w.add_scalar('sort/loss', loss.data[0], i*batch)

        if i % 100 == 0:
            print(t[0,:])
            print(y[0,:])
            print('-------')

        if False or i % plot_every == 0:

            means, sigmas, values = model.hyper(x)

            plt.cla()
            util.plot(means, sigmas, values, shape=(SHAPE[0], SHAPE[0]))
            plt.xlim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
            plt.ylim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
            plt.savefig('./sort/means{:04}.png'.format(i))

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Size of the input",
                        default=32, type=int)

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="The number of iterations (ie. the nr of batches).",
                        default=30000, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled",
                        default=512, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.01, type=float)

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

    parser.add_argument("-S", "--sigma-scale",
                        dest="sigma_scale",
                        help="Sigma scale.",
                        default=0.1, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(batch=options.batch_size,
        additional=options.additional, iterations=options.iterations, cuda=options.cuda,
        lr=options.lr, plot_every=options.plot_every, size=options.size, fv=options.fix_values,
        seed=options.seed, sigma_scale=options.sigma_scale)
