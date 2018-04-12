import hyper, gaussian
import torch, random, sys
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.functional import sigmoid
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import util, logging, time, gc
import numpy as np

from argparse import ArgumentParser

import psutil, os

from identity_experiment import go as igo

logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Simple experiment: learn the identity function from one tensor to another
"""
w = SummaryWriter()

def go(iterations=30000, additional=64, batch=4, size=32, cuda=False, plot_every=50,
       lr=0.01, fv=False, penalty_steps=10, sigma_scale=0.1, seed=0, repeats=10):
    print('.')

    plt.figure(figsize=(5,5))

    for penalty in np.logspace(-8, -1, num=penalty_steps):

        losses = []
        for r in range(repeats):
            loss = igo(iterations=iterations, additional=additional,
                                          batch=batch, size=size, cuda=cuda, plot_every = -1,
                                          lr=lr, fv=fv, sigma_penalty=float(penalty), sigma_scale=sigma_scale,
                                          seed=random.randint(0, 100000))
            losses.append(loss)

        print(penalty, losses)
        plt.plot([penalty] * repeats, losses, linewidth=0, marker='.')

    plt.gca().set_xscale("log")

    plt.savefig('losses.pdf')


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Size (nr of dimensions) of the input.",
                        default=32, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="The number of iterations (ie. the nr of batches).",
                        default=800, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled",
                        default=512, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-F", "--fix_values", dest="fix_values",
                        help="Whether to fix the values to 1.",
                        action="store_true")

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-S", "--sigma-scale",
                        dest="sigma_scale",
                        help="Sigma scale",
                        default=0.1, type=float)

    parser.add_argument("-P", "--penalty-steps",
                        dest="penalty_steps",
                        help="Sigma penalty steps",
                        default=10, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="Random seed.",
                        default=32, type=int)

    parser.add_argument("-R", "--repeats",
                        dest="repeats",
                        help="Repeats.",
                        default=10, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(batch=options.batch_size, size=options.size,
        additional=options.additional, iterations=options.iterations, cuda=options.cuda,
        lr=options.lr, fv=options.fix_values,
        sigma_scale=options.sigma_scale, penalty_steps=options.penalty_steps, seed=options.seed,
        repeats=options.repeats)
