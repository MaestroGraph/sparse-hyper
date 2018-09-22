import gaussian
import torch, random, sys
from torch.autograd import Variable
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

from argparse import ArgumentParser

import os
logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Simple experiment: learn the identity function from one tensor to another
"""
w = SummaryWriter()

def go(batch=4, cuda=False, plot_every=1000,
       lr=0.01, fv=False, sigma_scale=0.1, min_sigma=0.0, seed=0, reps=10, dot_every=100):

    sizes = [4, 8, 16, 32, 64, 128, 256]
    itss  = [40000, 80000, 160000, 320000, 640000, 640000, 640000]

    MARGIN = 0.1

    torch.manual_seed(seed)

    results = {}

    for si, size in enumerate(sizes):

        iterations = itss[si]
        ndots = iterations // dot_every

        additional = int(np.floor(np.log2(size)) * size)
        results[size] = np.zeros((2, reps, ndots))

        for reinforce in [True, False] if size < 32 else [False]:
            rf = 0 if not reinforce else 1

            print('Starting size {} with {} additional samples (reinforce={})'.format(size, additional, reinforce))

            for r in trange(reps):
                util.makedirs('./identity/{}/{}/{}'.format(reinforce, size, r))

                SHAPE = (size,)

                gaussian.PROPER_SAMPLING = size < 8
                model = gaussian.ParamASHLayer(
                    SHAPE, SHAPE, k=size, additional=additional,
                    sigma_scale=sigma_scale if not reinforce else size/7.0,
                    has_bias=False, fix_values=fv, min_sigma=min_sigma, reinforce=reinforce)

                if cuda:
                    model.cuda()

                optimizer = optim.Adam(model.parameters(), lr=lr)


                for i in trange(iterations):

                    x = torch.randn((batch,) + SHAPE)

                    if cuda:
                        x = x.cuda()
                    x = Variable(x)

                    optimizer.zero_grad()

                    if not reinforce:
                        y = model(x)

                        loss = F.mse_loss(y, x)

                    else:
                        y, dists, actions = model(x)

                        mloss = F.mse_loss(y, x, reduce=False).mean(dim=1)
                        rloss = - dists.log_prob(actions) * - mloss.data.unsqueeze(1).unsqueeze(1).expand_as(actions)

                        loss = rloss.mean()

                    loss.backward()

                    optimizer.step()

                    w.add_scalar('identity32/loss/{}/{}/{}'.format(reinforce, size, r), loss.item(), i*batch)

                    if i % dot_every == 0:
                        mse = F.mse_loss(y.data, x.data)

                        results[size][rf, r, i//dot_every] = mse.item()

                    if plot_every > 0 and i % plot_every == 0:
                        plt.figure(figsize=(7, 7))

                        means, sigmas, values = model.hyper(x)

                        plt.cla()
                        util.plot(means, sigmas, values, shape=(SHAPE[0], SHAPE[0]))
                        plt.xlim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
                        plt.ylim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))

                        plt.savefig('./identity/{}/{}/{}/means{:04}.pdf'.format(reinforce, size, r, i))

        np.save('results.{:03d}.np'.format(size), results)

    print('experiments finished')

    plt.figure(figsize=(5, 5))
    plt.clf()

    norm = mpl.colors.Normalize(vmin=min(sizes), vmax=max(sizes))
    cmap = plt.get_cmap('viridis')
    map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    for si, size in enumerate(sizes):
        color = map.to_rgba(size)
        res = results[size]
        iterations = itss[si]
        ndots = iterations // dot_every
        additional = int(np.floor(np.log2(size)) * size)

        for reinforce in [True, False] if size < 32 else [False]:
            rf = 0 if not reinforce else 1

            # print(reinforce, res[rf, :, :])
            # print(reinforce, np.mean(res[rf, :, :], axis=0))
            # print(reinforce, np.std(res[rf, :, :], axis=0))

            plt.errorbar(
                x=np.arange(ndots) * dot_every, y=np.mean(res[rf, :, :], axis=0), yerr=np.std(res[rf, :, :], axis=0),
                label='{} by {}, a={}, {}'.format(size, size, additional, 'reinforce' if reinforce else 'backprop'),
                color=color, linestyle='--' if reinforce else '-',  alpha=0.5 if reinforce else 1.0)

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

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Size (nr of dimensions) of the input.",
                        default=32, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

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

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(batch=options.batch_size, cuda=options.cuda,
        lr=options.lr, plot_every=options.plot_every, fv=options.fix_values,
        sigma_scale=options.sigma_scale, min_sigma=options.min_sigma, seed=options.seed,
       reps=options.reps, dot_every=options.dot_every)
