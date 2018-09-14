import gaussian_temp
import torch, random, sys
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.functional import sigmoid
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter
from gaussian import Bias

import matplotlib.pyplot as plt
import util, logging, time, gc
import numpy as np

from argparse import ArgumentParser

logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Template version of the identity experiment

"""
w = SummaryWriter()

class Layer(gaussian_temp.HyperLayer):

    def __init__(self, size=8, learn=1, additional=16):

        temp = torch.LongTensor(range(size)).unsqueeze(1).expand(size, 2)

        super().__init__(in_rank=1, out_size=(size,), temp_indices=temp, learn_cols=(learn,), additional=additional, subsample=None, bias_type=Bias.NONE)

        self.size = size
        self.params = nn.Parameter(torch.randn((size, 3)))

        self.sigma_scale = 0.1
        self.min_sigma = 0.0
        self.fix_values = True

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        batch_size = input.size()[0]

        # Replicate the parameters along the batch dimension
        res = self.params.unsqueeze(0).expand(batch_size, self.size, 3)

        means, sigmas, values = self.split_out(res, (self.size,))
        sigmas = sigmas * self.sigma_scale + self.min_sigma

        if self.fix_values:
            values = values * 0.0 + 1.0

        return means, sigmas, values


def go(iterations=30000, additional=64, batch=4, size=32, cuda=False, plot_every=50,
       lr=0.01, fv=False, sigma_scale=0.1, min_sigma=0.0, seed=0):

    SHAPE = (size,)
    MARGIN = 0.1

    torch.manual_seed(seed)

    nzs = util.prod(SHAPE)

    util.makedirs('./identity-temp/')

    params = None

    gaussian_temp.PROPER_SAMPLING = False
    model = Layer(size=size, additional=additional)

    if cuda:
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for i in trange(iterations):

        x = torch.zeros((batch,) + SHAPE) + (1.0/16.0)
        x = torch.bernoulli(x)
        if cuda:
            x = x.cuda()
        x = Variable(x)

        optimizer.zero_grad()

        y = model(x)

        loss = criterion(y, x)

        t0 = time.time()
        loss.backward()        # compute the gradients

        optimizer.step()

        w.add_scalar('identity-temp/loss', loss.data.item(), i*batch)

        if plot_every > 0 and i % plot_every == 0:
            plt.figure(figsize=(7, 7))

            means, sigmas, values = model.hyper(x)

            means, sigmas, values = means.data, sigmas.data, values.data

            template = model.temp_indices.float().unsqueeze(0).expand(means.size(0), means.size(1), 2)
            template[:, :, model.learn_cols] = means
            means = template

            plt.cla()
            util.plot1d(means[0], sigmas[0], values[0], shape=(SHAPE[0], SHAPE[0]))
            plt.xlim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
            plt.ylim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))

            plt.savefig('./identity-temp/means{:04}.png'.format(i))

    return float(loss.data[0])

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
                        default=3000, type=int)

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

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="Random seed.",
                        default=32, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(batch=options.batch_size, size=options.size,
        additional=options.additional, iterations=options.iterations, cuda=options.cuda,
        lr=options.lr, plot_every=options.plot_every, fv=options.fix_values,
        sigma_scale=options.sigma_scale, min_sigma=options.min_sigma, seed=options.seed)
