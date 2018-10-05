import gaussian, globalsampling
import torch, random, sys
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import util, logging, time, gc, math
import numpy as np

from scipy.stats import sem

from argparse import ArgumentParser

import os
logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Simple experiment: learn the identity function from one tensor to another
"""

def go(arg):

    MARGIN = 0.1

    iterations = arg.iterations if arg.iterations is not None else arg.size * 3000
    additional = arg.additional if arg.additional is not None else int(np.floor(np.log2(arg.size)) * arg.size)

    torch.manual_seed(arg.seed)

    ndots = iterations // arg.dot_every

    results = np.zeros((arg.reps, ndots))

    print('Starting size {} with {} additional samples (reinforce={})'.format(arg.size, additional, arg.reinforce))
    w = None
    for r in range(arg.reps):
        print('repeat {} of {}'.format(r, arg.reps))

        util.makedirs('./identity/{}'.format(r))
        util.makedirs('./runs/identity/{}'.format(r))

        if w is not None:
            w.close()
        w = SummaryWriter(log_dir='./runs/identity/{}/'.format(r))

        SHAPE = (arg.size,)

        gaussian.PROPER_SAMPLING = arg.size < 8

        if arg.global_sampling:
            model = globalsampling.ParamASHLayer(
                SHAPE, SHAPE, k=arg.size, additional=additional,
                sigma_scale=arg.sigma_scale,
                has_bias=False, fix_values=arg.fix_values, min_sigma=arg.min_sigma,
                relative_range=(arg.rr, arg.rr),
                rr_additional=arg.ca, subsample=arg.subsample)
        else:
            model = gaussian.ParamASHLayer(
                SHAPE, SHAPE, k=arg.size, additional=additional,
                sigma_scale=arg.sigma_scale if not arg.reinforce else arg.size/7.0,
                has_bias=False, fix_values=arg.fix_values, min_sigma=arg.min_sigma, reinforce=arg.reinforce,
                relative_range=None if arg.rr is None else (arg.rr, arg.rr),
                rr_additional=arg.ca)

        if arg.cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=arg.lr)


        for i in trange(iterations):

            x = torch.randn((arg.batch,) + SHAPE)

            if arg.cuda:
                x = x.cuda()
            x = Variable(x)


            if not arg.reinforce:


                if arg.subsample is None:
                    optimizer.zero_grad()

                    y = model(x)

                    loss = F.mse_loss(y, x)

                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.zero_grad()

                    # multiple forward/backward passes, accumulate gradient
                    seed = (torch.rand(1) * 100000).long().item()
                    for fr in range(0, arg.size, arg.subsample):
                        to = min(fr + arg.subsample, arg.size)

                        y = model(x, mrange=(fr, to), seed=seed)

                        loss = F.mse_loss(y, x)

                        loss.backward()
                    optimizer.step()

            else:

                optimizer.zero_grad()

                y, dists, actions = model(x)

                mloss = F.mse_loss(y, x, reduce=False).mean(dim=1)
                rloss = - dists.log_prob(actions) * - mloss.data.unsqueeze(1).unsqueeze(1).expand_as(actions)

                loss = rloss.mean()

                loss.backward()
                optimizer.step()

            w.add_scalar('identity/loss/', loss.item(), i*arg.batch)

            if i % arg.dot_every == 0:
                mse = F.mse_loss(y.data, x.data)
                results[r, i//arg.dot_every] = mse.item()

            if arg.plot_every > 0 and i % arg.plot_every == 0:
                plt.figure(figsize=(7, 7))

                means, sigmas, values = model.hyper(x)

                plt.cla()
                util.plot(means, sigmas, values, shape=(SHAPE[0], SHAPE[0]))
                plt.xlim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
                plt.ylim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))

                plt.savefig('./identity/{}/means{:04}.pdf'.format(r, i))

    np.save('results.{:03d}.{}'.format(arg.size, arg.reinforce), results)

    print('experiments finished')

    plt.figure(figsize=(10, 4))
    plt.clf()

    # print(reinforce, res[rf, :, :])
    # print(reinforce, np.mean(res[rf, :, :], axis=0))
    # print(reinforce, np.std(res[rf, :, :], axis=0))

    plt.errorbar(
        x=np.arange(ndots) * arg.dot_every, y=np.mean(results, axis=0), yerr=np.std(results, axis=0),
        label='{} by {}, a={}, {}'.format(arg.size, arg.size, additional, 'reinforce' if arg.reinforce else 'backprop'),
        linestyle='--' if arg.reinforce else '-',  alpha=0.7 if arg.reinforce else 1.0)

    plt.legend()

    ax = plt.gca()
    ax.set_ylim(bottom=0)
    ax.set_xlabel('iterations')
    ax.set_ylabel('mean-squared error')

    util.basic()

    plt.savefig('./identity/results.png')
    plt.savefig('./identity/results.pdf')


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="Size (nr of dimensions) of the input.",
                        default=None, type=int)

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Size (nr of dimensions) of the input.",
                        default=32, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled (automatically chosen if None)",
                        default=None, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-F", "--fix_values", dest="fix_values",
                        help="Whether to fix the values to 1.",
                        action="store_true")

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.005, type=float)

    parser.add_argument("-S", "--sigma-scale",
                        dest="sigma_scale",
                        help="Sigma scale",
                        default=0.1, type=float)

    parser.add_argument("-M", "--min_sigma",
                        dest="min_sigma",
                        help="Minimum variance for the components.",
                        default=0.0, type=float)

    parser.add_argument("-p", "--plot-every",
                        dest="plot_every",
                        help="Plot every x iterations",
                        default=50, type=int)

    parser.add_argument("-Q", "--subsample",
                        dest="subsample",
                        help="The number of index tuples to subsample for learning",
                        default=None, type=int)

    parser.add_argument("-d", "--dot-every",
                        dest="dot_every",
                        help="A dot in the graph for every x iterations",
                        default=100, type=int)

    parser.add_argument("-R", "--repeats",
                        dest="reps",
                        help="Number of repeats.",
                        default=10, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="Random seed.",
                        default=32, type=int)

    parser.add_argument("-C", "--chunk-size",
                        dest="rr",
                        help="Size of the sampling region around the index tuple (if None, only uniform sampling is used).",
                        default=None, type=int)

    parser.add_argument("-A", "--chunk-additional",
                        dest="ca",
                        help="Number of points to sample for sampling region.",
                        default=None, type=int)

    parser.add_argument("-B", "--use-reinforce", dest="reinforce",
                        help="Use reinforce instead of the backprop approach.",
                        action="store_true")

    parser.add_argument("-G", "--use-global-sampling", dest="global_sampling",
                        help="Use the global sampling approach.",
                        action="store_true")

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(options)
