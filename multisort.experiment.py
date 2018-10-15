import sort
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

import torchvision
from torch.utils.data import TensorDataset, DataLoader

import os, sys

from gaussian import HyperLayer

logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Experiment: learn to sort numbers consisting of multiple MNIST digits.

"""
tbw = SummaryWriter()

util.DEBUG = False
BUCKET_SIGMA = 0.05

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    [s.set_visible(False) for s in axes.spines.values()]
    axes.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)


def gen(b, data, labels, size, digits):

    n = data.size(0)

    total = b * size * digits
    inds = random.choices(range(n), k=total)

    x   = data[inds, :, :, :]
    l = labels[inds]

    x = x.view(b, size, digits, 1, 28, 28)
    l = l.view(b, size, digits)

    power = 10 ** torch.arange(digits, dtype=torch.long)
    l = (l * power).sum(dim=2)

    _, idx = l.sort(dim=1)

    t = x.gather(dim=1, index=idx[:, :, None, None, None, None].expand(b, size, digits, 1, 28, 28))

    return x, t, l

def plotn(data, ax):

    n = data.size(0)

    for i in range(n):
        im = data[i].data.cpu().numpy()
        ax.imshow(im, extent=(n-i-1, n-i, 0, 1), cmap='gray_r')

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)

    ax.axhline()

def go(arg):
    """

    :param arg:
    :return:
    """

    torch.manual_seed(arg.seed)
    np.random.seed(arg.seed)
    random.seed(arg.seed)

    torch.set_printoptions(precision=10)

    """
    Load and organize the data
    """
    trans = torchvision.transforms.ToTensor()
    if arg.final:
        train = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=trans)
        trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch, shuffle=True, num_workers=2)

        test = torchvision.datasets.MNIST(root=arg.data, train=False, download=True, transform=trans)
        testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch, shuffle=False, num_workers=2)

    else:
        NUM_TRAIN = 45000
        NUM_VAL = 5000
        total = NUM_TRAIN + NUM_VAL

        train = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=trans)

        trainloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
        testloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

    shape = (1, 28, 28)
    num_classes = 10

    xbatches = []
    lbatches = []
    for xbatch, lbatch in trainloader:
        xbatches.append(xbatch)
        lbatches.append(lbatch)

    data   = torch.cat(xbatches, dim=0)
    labels = torch.cat(lbatches, dim=0)

    if arg.limit is not None:
        data = data[:arg.limit]
        labels = labels[:arg.limit]

    xbatches = []
    lbatches = []
    for xbatch, lbatch in testloader:
        xbatches.append(xbatch)
        lbatches.append(lbatch)

    data_test   = torch.cat(xbatches, dim=0)
    labels_test = torch.cat(lbatches, dim=0)

    ndots = arg.iterations // arg.dot_every

    results = np.zeros((arg.reps, ndots))

    for r in range(arg.reps):
        print('starting {} out of {} repetitions'.format(r, arg.reps))
        util.makedirs('./multisort/{}'.format( r))

        model = sort.SortLayer(arg.size, additional=arg.additional, sigma_scale=arg.sigma_scale,
                               sigma_floor=arg.min_sigma, certainty=arg.certainty)

        # - channel sizes
        c1, c2, c3 = 16, 64, 128
        h1, h2, out= 256, 128, 8

        per_digit = nn.Sequential(
            nn.Conv2d(1, c1, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c1, c1, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c1, c1, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(c1),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c1, c2, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c2, c2, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c2, c2, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(c2),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c2, c3, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c3, c3, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c3, c3, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(c3),
            nn.MaxPool2d((2, 2)),
            util.Flatten(),
            nn.Linear(9 * c3, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, out)# , nn.BatchNorm1d(1),
        )

        hidden = 256

        tokeys = nn.Sequential(
            util.Lambda(lambda x : x.view(arg.batch * arg.size * arg.digits, 1, 28, 28)),
            per_digit,
            util.Lambda(lambda x: x.view(arg.batch * arg.size, arg.digits * out)),
            nn.Linear(out * arg.digits, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
            util.Lambda(lambda x: x.view(arg.batch, arg.size))
        )

        if arg.cuda:
            model.cuda()
            tokeys.cuda()

        optimizer = optim.Adam(list(model.parameters()) + list(tokeys.parameters()), lr=arg.lr)

        for i in trange(arg.iterations):

            x, t, l = gen(arg.batch, data, labels, arg.size, arg.digits)

            if arg.cuda:
                x, t = x.cuda(), t.cuda()

            x, t = Variable(x), Variable(t)

            optimizer.zero_grad()

            keys = tokeys(x)

            # keys = keys * 0.0 + l

            x = x.view(arg.batch, arg.size, -1)
            t = t.view(arg.batch, arg.size, -1)

            ys, ts, keys = model(x, keys=keys, target=t)

            if not arg.use_intermediates:
                # just compare the output to the target
                loss = util.xent(ys[-1], t).mean()
            else:
                # compare the output to the back-sorted target at each step
                loss = 0.0
                loss = loss + util.xent(ys[0], ts[0]).mean()
                loss = loss + util.xent(ts[-1], ts[-1]).mean()

                # average over the buckets
                for d in range(1, len(ys)-1):
                    numbuckets = 2 ** d
                    bucketsize = arg.size // numbuckets

                    xb = ys[d][:, None, :, :].view(arg.batch, numbuckets, bucketsize, -1)
                    tb = ts[d][:, None, :, :].view(arg.batch, numbuckets, bucketsize, -1)

                    xb = xb.mean(dim=2)
                    tb = tb.mean(dim=2)

                    loss = loss + util.xent(xb, tb).mean() * bucketsize

            loss.backward()

            optimizer.step()

            tbw.add_scalar('multisort/loss/{}/{}'.format(arg.size, r), loss.data.item(), i*arg.batch)
            # tbw.add_scalar('multisort/cert/{}/{}'.format(arg.size, r), model.certainty, i*arg.batch)

            # Plot intermediate results, and targets
            if i % arg.plot_every == 0 and False:

                optimizer.zero_grad()

                x, t, l = gen(arg.batch, data, labels, arg.size, arg.digits)

                if arg.cuda:
                    x, t = x.cuda(), t.cuda()

                x, t = Variable(x), Variable(t)

                keys = tokeys(x)

                x = x.view(arg.batch, arg.size, -1)
                t = t.view(arg.batch, arg.size, -1)

                ys, ts, _ = model(x, keys=keys, target=t)

                b, n, s = ys[0].size()

                for d in range(1, len(ys) - 1):
                    numbuckets = 2 ** d
                    bucketsize = arg.size // numbuckets

                    xb = ys[d][:, None, :, :].view(arg.batch, numbuckets, bucketsize, s)
                    tb = ts[d][:, None, :, :].view(arg.batch, numbuckets, bucketsize, s)

                    xb = xb.mean(dim=2, keepdim=True)\
                        .expand(arg.batch, numbuckets, bucketsize, s)\
                        .contiguous().view(arg.batch, n, s)
                    tb = tb.mean(dim=2, keepdim=True)\
                        .expand(arg.batch, numbuckets, bucketsize, s)\
                        .contiguous().view(arg.batch, n, s)

                    ys[d] = xb
                    ts[d] = tb

                md = int(np.log2(arg.size))
                plt.figure(figsize=(arg.size*2, md+1))

                c = 1
                for row in range(md + 1):
                    for col in range(arg.size*2):
                        ax = plt.subplot(md+1, arg.size*2, c)

                        images = ys[row] if col < arg.size else ts[row]
                        im = images[0].view(arg.size, 28, 28)[col%arg.size].data.cpu().numpy()

                        ax.imshow(im, cmap='gray_r')
                        clean(ax)

                        c += 1

                plt.savefig('./mnist-sort/{}/intermediates.{:04}.pdf'.format(r, i))

            # Plot the progress
            if i % arg.plot_every == 0:

                optimizer.zero_grad()

                x, t, l = gen(arg.batch, data, labels, arg.size, arg.digits)

                if arg.cuda:
                    x, t = x.cuda(), t.cuda()

                x, t = Variable(x), Variable(t)

                keys = tokeys(x)
                keys.retain_grad()

                x = x.view(arg.batch, arg.size, -1)
                t = t.view(arg.batch, arg.size, -1)

                yt, _ = model(x, keys=keys, train=True)

                loss = F.mse_loss(yt, t)  # compute the loss
                loss.backward()

                yi, _ = model(x, keys=keys, train=False)

                input  = x[0].view(arg.size, arg.digits, 28, 28)
                target = t[0].view(arg.size, arg.digits, 28, 28)
                output_inf   = yi[0].view(arg.size, arg.digits, 28, 28)
                output_train = yt[0].view(arg.size, arg.digits, 28, 28)

                plt.figure(figsize=(arg.size*3*arg.digits, 4*3))
                for col in range(arg.size):

                    ax = plt.subplot(4, arg.size, col + 1)
                    plotn(target[col], ax)
                    clean(ax)

                    if col == 0:
                        ax.set_ylabel('target')

                    ax = plt.subplot(4, arg.size, col + arg.size + 1)
                    plotn(input[col], ax)
                    clean(ax)
                    ax.set_xlabel( '{:.2}, {:.2}'.format(keys[0, col], - keys.grad[0, col] ) )

                    if col == 0:
                        ax.set_ylabel('input')

                    ax = plt.subplot(4, arg.size, col + arg.size * 2 + 1)
                    plotn(output_inf[col], ax)
                    clean(ax)

                    if col == 0:
                        ax.set_ylabel('inference')

                    ax = plt.subplot(4, arg.size, col + arg.size * 3 + 1)
                    plotn(output_train[col], ax)
                    clean(ax)

                    if col == 0:
                        ax.set_ylabel('training')

                plt.savefig('./multisort/{}/mnist.{:04}.pdf'.format(r, i))

            if i % arg.dot_every == 0:
                """
                Compute the accuracy
                """
                NUM = 10_000
                tot = 0.0
                correct = 0.0
                with torch.no_grad():

                    losses = []
                    for ii in range(NUM//arg.batch):
                        x, t, l = gen(arg.batch, data, labels, arg.size, arg.digits)

                        if arg.cuda:
                            x, t, l = x.cuda(), t.cuda(), l.cuda()

                        x, t, l = Variable(x), Variable(t), Variable(l)

                        keys = tokeys(x)

                        # Sort the keys, and sort the labels, and see if the resulting indices match
                        _, gold = torch.sort(l, dim=1)
                        _, mine = torch.sort(keys, dim=1)

                        tot += x.size(0)
                        correct += ((gold != mine).sum(dim=1) == 0).sum().item()

                    print('acc', correct/tot)

                    results[r, i//arg.dot_every] = np.mean(correct/tot)

                    tbw.add_scalar('multisort/testloss/{}/{}'.format(arg.size, r), correct/tot, i * arg.batch)

    np.save('results.{}.np'.format(arg.size), results)
    print('experiments finished')

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    if results.shape[0] > 1:
        ax.errorbar(x=np.arange(ndots) * arg.dot_every, y=np.mean(results[:, :], axis=0),
                        yerr=np.std(results[:, :], axis=0),
                        label='size {0}x{0}, r={1}'.format(arg.size, arg.reps))
    else:
        ax.plot(np.arange(ndots) * arg.dot_every, np.mean(results[:, :], axis=0),
                        label='size {0}x{0}'.format(arg.size))

    ax.legend()

    util.basic(ax)

    ax.spines['bottom'].set_position('zero')
    ax.set_ylim(0.0, 1.0)
#    ax.set_xlim(0.0, 100.0)

    plt.xlabel('iterations')
    plt.ylabel('error')

    plt.savefig('./quicksort/result.png')
    plt.savefig('./quicksort/result.pdf')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Dimensionality of the input.",
                        default=128, type=int)

    parser.add_argument("-w", "--width",
                        dest="digits",
                        help="Number of digits in each number sampled.",
                        default=3, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="The batch size.",
                        default=128, type=int)

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="Number of iterations (in batches).",
                        default=8000, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled globally",
                        default=2, type=int)

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

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="Random seed.",
                        default=32, type=int)

    parser.add_argument("-d", "--dot-every",
                        dest="dot_every",
                        help="How many iterations per dot in the loss curves.",
                        default=1000, type=int)

    parser.add_argument("-D", "--data",
                        dest="data",
                        help="Data ditectory.",
                        default='./data', type=str)

    parser.add_argument("-S", "--sigma-scale",
                        dest="sigma_scale",
                        help="Sigma scale.",
                        default=0.1, type=float)

    parser.add_argument("-C", "--certainty",
                        dest="certainty",
                        help="Certainty: scaling factor in the bucketing computation.",
                        default=10.0, type=float)

    parser.add_argument("-R", "--repeats",
                        dest="reps",
                        help="Number of repeats.",
                        default=10, type=int)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Sigma floor (minimum sigma value).",
                        default=0.0, type=float)

    parser.add_argument("-L", "--limit",
                        dest="limit",
                        help="Limit on the nr ofexamples per class (for debugging).",
                        default=None, type=int)

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set.",
                        action="store_true")

    parser.add_argument("-I", "--intermediates",
                        dest="use_intermediates",
                        help="Whether to backwards-sort the target to provide a loss at every step.",
                        action="store_true")


    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(options)
