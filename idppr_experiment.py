import gaussian
import torch, random, sys
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.functional import sigmoid
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

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

    sizes = [4, 8, 16, 32, 64, 128] # 256, 512, 1024]
    itss  = [200, 400, 80000, 320000, 640000, 640000]

    MARGIN = 0.1

    torch.manual_seed(seed)

    results = {}

    for si, size in enumerate(sizes):

        iterations = itss[si]
        ndots = iterations // dot_every

        results[size] = np.zeros((reps, ndots))

        additional = int(np.floor(np.log2(size)) * size)

        print('Starting size {} with {} additional samples '.format(size, additional))

        for r in trange(reps):
            util.makedirs('./identity/{}/{}'.format(size, r))

            SHAPE = (size,)

            gaussian.PROPER_SAMPLING = size < 8
            model = gaussian.ParamASHLayer(SHAPE, SHAPE, k=size, additional=additional, sigma_scale=sigma_scale, has_bias=False, fix_values=fv, min_sigma=min_sigma)

            if cuda:
                model.cuda()

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)


            for i in trange(iterations):

                x = torch.randn((batch,) + SHAPE)

                if cuda:
                    x = x.cuda()
                x = Variable(x)

                optimizer.zero_grad()

                y = model(x)

                loss = criterion(y, x)

                loss.backward()

                optimizer.step()

                w.add_scalar('identity32/loss/{}/{}'.format(size, r), loss.item(), i*batch)

                if i % dot_every == 0:
                    results[size][r, i//dot_every] = loss.item()

                if plot_every > 0 and i % plot_every == 0:
                    plt.figure(figsize=(7, 7))

                    means, sigmas, values = model.hyper(x)

                    plt.cla()
                    util.plot(means, sigmas, values, shape=(SHAPE[0], SHAPE[0]))
                    plt.xlim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
                    plt.ylim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))

                    plt.savefig('./identity/{}/{}/means{:04}.pdf'.format(size, r, i))


        np.save('results.{:03d}.np'.format(size), results)

    print('experiments finished')


    plt.figure(figsize=(5, 5))
    plt.clf()

    for si, size in enumerate(sizes):
        iterations = itss[si]
        res = results[size]
        ndots = iterations // dot_every

        additional = int(np.floor(np.log2(size)) * size)

        print(sem(res[:, :], axis=0))
        plt.errorbar(x=np.arange(ndots) * dot_every, y=np.mean(results[size][:, :], axis=0), yerr=sem(results[size][:, :], axis=0), label='{} by {}, a={}'.format(size, size, additional))
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
       reps=options.reps)
