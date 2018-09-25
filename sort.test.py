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

from argparse import ArgumentParser

from gaussian import HyperLayer

logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Check if the sort function is learnable directly
"""
w = SummaryWriter()

def go(arg):

    HIDDENBIG = arg.size * arg.size ** 2


    torch.manual_seed(arg.seed)
    activation = nn.ReLU()

    layers = []
    layers.append(nn.Linear(arg.size, HIDDENBIG))
    layers.append(activation)

    for _ in range(arg.depth):
        layers.append(nn.Linear(HIDDENBIG, HIDDENBIG, bias=True))
        # layers.append(nn.BatchNorm1d(HIDDENBIG))
        layers.append(activation)

    layers.append(nn.Linear(HIDDENBIG, arg.size))
    layers.append(nn.Sigmoid())
    layers.append(util.Lambda(lambda x : x * arg.size))

    model = nn.Sequential(*layers)

    if arg.cuda:
        model.cuda()

    errors = []

    optimizer = optim.Adam(model.parameters(), lr=arg.lr)

    plt.figure(figsize=(16, 8))

    for i in trange(arg.iterations):

        x = torch.randn((arg.batch, arg.size))
        t = x.sort()[1].float()

        if arg.cuda:
            x, t = x.cuda(), t.cuda()

        x, t = Variable(x), Variable(t)

        optimizer.zero_grad()

        y = model(x)

        loss = F.mse_loss(y, t)

        if i % 1000 == 0:
            # Compute accuracy estimate

            correct = 0
            tot = 0
            for ii in range(10000 // arg.batch):
                x = torch.randn((arg.batch, arg.size))
                t = x.sort()[1].float()

                if arg.cuda:
                    x, t = x.cuda(), t.cuda()
                x, t = Variable(x), Variable(t)

                y = model(x).round()

                # if ii == 0:
                #     print(y[:3].long())
                #     print(t[:3].long())
                #     print(y[:3].long() != t[:3].long())
                #     print((y[:3].long()!=t[:3].long()).sum(dim=1))

                neq = (y.long()!=t.long()).sum(dim=1)

                correct += (neq == 0).sum().item()
                tot += x.size(0)
            print('acc', correct/tot)

            errors.append(1 - (correct/tot))

        t0 = time.time()
        loss.backward()        # compute the gradients

        optimizer.step()

        w.add_scalar('sort-direct/loss', loss.data[0], i*arg.batch)

        if i % 1000 == 0:
            plt.clf()
            plt.plot(errors)

            util.clean()
            plt.ylim(0, 1)
            plt.savefig('sorting-test.errors.png')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Size of the input",
                        default=5, type=int)

    parser.add_argument("-d", "--depth",
                        dest="depth",
                        help="Depth of the sorting network",
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

    go(options)
