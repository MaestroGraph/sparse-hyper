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

import os

from gaussian import HyperLayer

logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Experiment: learn a mapping from a random x, to x sorted.

"""
tbw = SummaryWriter()

util.DEBUG = False
BUCKET_SIGMA = 0.05

SIZE = 4

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    [s.set_visible(False) for s in axes.spines.values()]
    axes.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)

def gen(b, data):

    t = torch.zeros(b, SIZE, 1, 28, 28)

    # Select random digits
    for i in range(SIZE):
        sample = random.sample(data[i], k=b)
        t[:, i, :, :, :] = torch.cat(sample, dim=0)

    x = t.clone()

    # Shuffle
    for bi in range(b):
        perm = torch.randperm(SIZE)
        x[bi] = x[bi, perm, :, :, :]

    return x, t

def go(arg):
    """

    :param arg:
    :return:
    """

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

    train = {label: [] for label in range(10)}

    for inps, labels in trainloader:
        b, c, h, w = inps.size()
        for i in range(b):
            image = inps[i:i+1, :, :, :]
            label = labels[i].item()
            train[label].append(image)

    # train = {label: torch.cat(imgs, dim=0) for label, imgs in train}

    test = {label: [] for label in range(10)}
    for inps, labels in trainloader:
        b, c, h, w = inps.size()
        for i in range(b):
            image = inps[i:i+1, :, :, :]
            label = labels[i].item()
            test[label].append(image)

    # train = {label: torch.cat(imgs, dim=0) for label, imgs in train}
    del b, c, h, w

    torch.manual_seed(arg.seed)

    ndots = arg.iterations // arg.dot_every

    results = np.zeros((arg.reps, ndots))

    for r in range(arg.reps):
        print('starting {} out of {} repetitions'.format(r, arg.reps))
        util.makedirs('./mnist-sort/{}'.format( r))

        model = sort.SortLayer(SIZE, additional=arg.additional, sigma_scale=arg.sigma_scale, sigma_floor=arg.min_sigma)

        bottom = nn.Linear(28*28, 256)
        # tokeys = nn.Sequential(
        #     util.Flatten(),
        #     bottom, nn.ReLU(),
        #     nn.Linear(256, 64), nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        # - channel sizes
        c1, c2, c3 = 4, 6, 8
        h1, h2 = 64, 32

        tokeys = nn.Sequential(
            nn.Conv2d(1, c1, (3, 3), padding=1), nn.ReLU(),
            # nn.Conv2d(c1, c1, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c1, c1, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(c1),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c1, c2, (3, 3), padding=1), nn.ReLU(),
            # nn.Conv2d(c2, c2, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c2, c2, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(c2),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c2, c3, (3, 3), padding=1), nn.ReLU(),
            # nn.Conv2d(c3, c3, (3, 3), padding=1), nn.ReLU(),
            nn.Conv2d(c3, c3, (3, 3), padding=1), nn.ReLU(),
            nn.BatchNorm2d(c3),
            nn.MaxPool2d((2, 2)),
            util.Flatten(),
            nn.Linear(9 * c3, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1), nn.BatchNorm1d(1),
        )

        if arg.cuda:
            model.cuda()
            tokeys.cuda()

        optimizer = optim.Adam(list(model.parameters()) + list(tokeys.parameters()), lr=arg.lr)

        for i in trange(arg.iterations):

            if i > 3000:
                util.DEBUG = True

            x, t = gen(arg.batch, train) # torch.randn((arg.batch_size,) + SHAPE)

            t, idxs = x.sort(dim=1)

            if arg.cuda:
                x, t = x.cuda(), t.cuda()
            x, t = Variable(x), Variable(t)

            optimizer.zero_grad()

            keys = tokeys(x.view(arg.batch * SIZE, 1, 28, 28))
            keys = keys.view(arg.batch, SIZE)

            x = x.view(arg.batch, SIZE, -1)
            t = t.view(arg.batch, SIZE, -1)

            y, _ = model(x, keys=keys)

            loss = F.mse_loss(y, t) # compute the loss

            loss.backward()

            optimizer.step()

            tbw.add_scalar('mnist-sort/loss/{}/{}'.format(arg.size, r), loss.data.item(), i*arg.batch)

            # Compute test set loss
            if i % arg.plot_every == 0:
                with torch.no_grad():

                    x, t = gen(arg.batch, test)
                    if arg.cuda:
                        x, t = x.cuda(), t.cuda()

                    x, t = Variable(x), Variable(t)

                    keys = tokeys(x.view(arg.batch * SIZE, 1, 28, 28))
                    # keys = tokeys(x.view(arg.batch * SIZE, -1))
                    keys = keys.view(arg.batch, SIZE)

                    x = x.view(arg.batch, SIZE, -1)
                    t = t.view(arg.batch, SIZE, -1)

                    yi, _ = model(x, keys=keys, train=False)
                    yt, _ = model(x, keys=keys, train=True)

                    input  = x[0].view(SIZE, 28, 28).data.cpu().numpy()
                    target = t[0].view(SIZE, 28, 28).data.cpu().numpy()
                    output_inf   = yi[0].view(SIZE, 28, 28).data.cpu().numpy()
                    output_train = yt[0].view(SIZE, 28, 28).data.cpu().numpy()

                    plt.figure(figsize=(SIZE, 4))
                    for col in range(SIZE):
                        ax = plt.subplot(4, SIZE, col + 1)
                        ax.imshow(input[col], cmap='gray_r')
                        clean(ax)

                        if col == 0:
                            ax.set_ylabel('input')

                        ax = plt.subplot(4, SIZE, col + SIZE + 1)
                        ax.imshow(target[col], cmap='gray_r')
                        clean(ax)

                        if col == 0:
                            ax.set_ylabel('target')

                        ax = plt.subplot(4, SIZE, col + SIZE * 2 + 1)
                        ax.imshow(output_inf[col], cmap='gray_r')
                        clean(ax)

                        if col == 0:
                            ax.set_ylabel('inference')

                        ax = plt.subplot(4, SIZE, col + SIZE * 3 + 1)
                        ax.imshow(output_train[col], cmap='gray_r')
                        clean(ax)

                        if col == 0:
                            ax.set_ylabel('training')

                    plt.savefig('./mnist-sort/mnist.{:04}.pdf'.format(i))

            if i % arg.dot_every == 0:
                # print('gradient ', bottom.weight.grad.mean(), bottom.weight.grad.var())

                with torch.no_grad():

                    losses = []
                    for ii in range(10000//arg.batch):
                        x, t = gen(arg.batch, test)

                        if arg.cuda:
                            x, t = x.cuda(), t.cuda()

                        x, t = Variable(x), Variable(t)

                        keys = tokeys(x.view(arg.batch * SIZE, 1, 28, 28))
                        #keys = tokeys(x.view(arg.batch * SIZE, -1))
                        keys = keys.view(arg.batch, SIZE)

                        x = x.view(arg.batch, SIZE, -1)
                        t = t.view(arg.batch, SIZE, -1)

                        y, _ = model(x, keys=keys, train=False)

                        loss = F.mse_loss(y, t)  # compute the loss

                        losses.append(loss.item())

                    print('loss', np.mean(losses))

                    results[r, i//arg.dot_every] = np.mean(losses)

                    tbw.add_scalar('mnist-sort/testloss/{}/{}'.format(arg.size, r), np.mean(losses), i * arg.batch)

    np.save('results.{}.np'.format(arg.size), results)
    print('experiments finished')

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    if results.shape[0] > 1:
        ax.errorbar(x=np.arange(ndots) * arg.dot_every, y=np.mean(results[:, :], axis=0),
                        yerr=np.std(results[:, :], axis=0),
                        label='size {0}x{0}, r={1}'.format(SIZE, arg.reps))
    else:
        ax.plot(np.arange(ndots) * arg.dot_every, np.mean(results[:, :], axis=0),
                        label='size {0}x{0}'.format(SIZE))

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
                        default=8, type=int)

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

    parser.add_argument("-R", "--repeats",
                        dest="reps",
                        help="Number of repeats.",
                        default=10, type=int)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Sigma floor (minimum sigma value).",
                        default=0.0, type=float)

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set.",
                        action="store_true")

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(options)
