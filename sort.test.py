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
Check if the sort function is learnable directly
"""
w = SummaryWriter()

def go(iterations=30000, batch=4, size=32, cuda=False, lr=0.01, seed=0):

    HIDDENBIG = size * size ** 2
    HIDDEN = size * 3

    torch.manual_seed(seed)

    model = nn.Sequential(
        nn.Linear(size, HIDDENBIG),
        nn.ReLU(),
        nn.Linear(HIDDENBIG, HIDDENBIG),
        nn.ReLU(),
        nn.Linear(HIDDENBIG, HIDDENBIG),
        nn.ReLU(),
        nn.Linear(HIDDENBIG, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, size),
    )

    if cuda:
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in trange(iterations):

        x = (torch.rand((batch, size)) * 64.0).floor() / 32.0 - 1.0

        t = x.sort()[1].float()

        if cuda:
            x, t = x.cuda(), t.cuda()

        x, t = Variable(x), Variable(t)

        optimizer.zero_grad()

        y = model(x)

        loss = criterion(y, t)

        if i % 1000 == 0:
            print((y-t)[:10, :])

        t0 = time.time()
        loss.backward()        # compute the gradients

        optimizer.step()

        w.add_scalar('sort-direct/loss', loss.data[0], i*batch)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=512, type=int)

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Size of the input",
                        default=5, type=int)

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="The number of iterations (ie. the nr of batches).",
                        default=100000, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0005, type=float)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="Random seed.",
                        default=32, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(batch=options.batch_size,
        iterations=options.iterations, cuda=options.cuda,
        lr=options.lr, size=options.size,
        seed=options.seed)
