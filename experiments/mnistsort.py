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
Experiment learn to sort single mnist digits. See multisort.experiment.py for the experiment reported in the paper.
"""
tbw = SummaryWriter()

util.DEBUG = False
BUCKET_SIGMA = 0.05

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    [s.set_visible(False) for s in axes.spines.values()]
    axes.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)


def gen(b, data, size):

    t = torch.zeros(b, size, 1, 28, 28)
    l = torch.zeros(b, size)

    # Select random digits
    for i in range(size):
        sample = random.choices(data[i], k=b)

        t[:, i, :, :, :] = torch.cat(sample, dim=0)
        l[:, i] = i

    x = t.clone()

    # Shuffle
    for bi in range(b):
        perm = torch.randperm(size)

        x[bi] = x[bi, perm, :, :, :]
        l[bi] =l[bi, perm]

    return x, t, l

def go(arg):
    """

    :param arg:
    :return:
    """

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

    train = {label: [] for label in range(10)}

    for inps, labels in trainloader:
        b, c, h, w = inps.size()
        for i in range(b):
            image = inps[i:i+1, :, :, :]
            label = labels[i].item()
            train[label].append(image)

    if arg.limit is not None:
        train = {label: imgs[:arg.limit] for label, imgs in train.items()}

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
    np.random.seed(arg.seed)
    random.seed(arg.seed)

    ndots = arg.iterations // arg.dot_every

    results = np.zeros((arg.reps, ndots))

    for r in range(arg.reps):
        print('starting {} out of {} repetitions'.format(r, arg.reps))
        util.makedirs('./mnist-sort/{}'.format( r))

        model = sort.SortLayer(arg.size, additional=arg.additional, sigma_scale=arg.sigma_scale,
                               sigma_floor=arg.min_sigma, certainty=arg.certainty)

        # bottom = nn.Linear(28*28, 32, bias=False)
        # bottom.weight.retain_grad()

        # top = nn.Linear(32, 1)
        # top.weight.retain_grad()

        # tokeys = nn.Sequential(
        #     util.Flatten(),
        #     bottom, nn.ReLU(),
        #     nn.Linear(32, 1)# , nn.BatchNorm1d(1)
        # )

        # - channel sizes
        c1, c2, c3 = 16, 64, 128
        h1, h2 = 256, 128

        tokeys = nn.Sequential(
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
            nn.Linear(h2, 1)# , nn.BatchNorm1d(1),
        )

        if arg.cuda:
            model.cuda()
            tokeys.cuda()

        optimizer = optim.Adam(list(model.parameters()) + list(tokeys.parameters()), lr=arg.lr)

        for i in trange(arg.iterations):

            x, t, l = gen(arg.batch, train, arg.size)

            if arg.cuda:
                x, t = x.cuda(), t.cuda()

            x, t = Variable(x), Variable(t)

            optimizer.zero_grad()

            keys = tokeys(x.view(arg.batch * arg.size, 1, 28, 28))
            keys = keys.view(arg.batch, arg.size)

            # keys = keys * 0.0 + l

            keys.retain_grad()

            x = x.view(arg.batch, arg.size, -1)
            t = t.view(arg.batch, arg.size, -1)

            ys, ts, keys = model(x, keys=keys, target=t)

            if arg.loss == 'plain':
                # just compare the output to the target
                # loss = F.mse_loss(ys[-1], t) # compute the loss
                # loss = F.binary_cross_entropy(ys[-1].clamp(0, 1), t.clamp(0, 1))
                loss = util.xent(ys[-1], t).mean()
            elif arg.loss == 'means':
                # compare the output to the back-sorted target at each step
                loss = 0.0
                loss = loss + util.xent(ys[0], ts[0]).mean()
                loss = loss + util.xent(ts[-1], ts[-1]).mean()

                for d in range(1, len(ys)-1):
                    numbuckets = 2 ** d
                    bucketsize = arg.size // numbuckets

                    xb = ys[d][:, None, :, :].view(arg.batch, numbuckets, bucketsize, -1)
                    tb = ts[d][:, None, :, :].view(arg.batch, numbuckets, bucketsize, -1)

                    xb = xb.mean(dim=2)
                    tb = tb.mean(dim=2)

                    loss = loss + util.xent(xb, tb).mean() * bucketsize

            elif 'separate':
                # compare the output to the back-sorted target at each step
                loss = 0.0
                loss = loss + util.xent(ts[-1], ts[-1]).mean()

                for d in range(0, len(ys)):
                    loss = loss + util.xent(ys[d], ts[d]).mean()

            else:
                raise Exception('Loss {} not recognized.'.format(arg.loss))

            loss.backward()

            optimizer.step()

            tbw.add_scalar('mnist-sort/loss/{}/{}'.format(arg.size, r), loss.data.item(), i*arg.batch)

            # Plot intermediates, and targets
            if i % arg.plot_every == 0:

                optimizer.zero_grad()

                x, t, l = gen(arg.batch, train, arg.size)

                if arg.cuda:
                    x, t = x.cuda(), t.cuda()

                x, t = Variable(x), Variable(t)

                keys = tokeys(x.view(arg.batch * arg.size, 1, 28, 28))
                keys = keys.view(arg.batch, arg.size)

                # keys = keys * 0.0 + l

                x = x.view(arg.batch, arg.size, -1)
                t = t.view(arg.batch, arg.size, -1)

                ys, ts, _ = model(x, keys=keys, target=t)

                b, n, s = ys[0].size()

                if arg.loss == 'means':

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

                        ax.imshow(im, cmap= 'bone_r' if col < arg.size else 'pink_r')

                        clean(ax)

                        c += 1

                plt.figtext(0.3, 0.95, "input", va="center", ha="center", size=15)
                plt.figtext(0.7, 0.95, "target", va="center", ha="center", size=15)

                plt.savefig('./mnist-sort/{}/intermediates.{:04}.pdf'.format(r, i))

            # Plot the progress
            if i % arg.plot_every == 0:

                optimizer.zero_grad()

                x, t, l = gen(arg.batch, train, arg.size)

                if arg.cuda:
                    x, t = x.cuda(), t.cuda()

                x, t = Variable(x), Variable(t)

                keys = tokeys(x.view(arg.batch * arg.size, 1, 28, 28))
                keys = keys.view(arg.batch, arg.size)
                # keys = keys * 0.01 + l
                keys.retain_grad()

                x = x.view(arg.batch, arg.size, -1)
                t = t.view(arg.batch, arg.size, -1)

                yt, _ = model(x, keys=keys, train=True)

                loss = F.mse_loss(yt, t)  # compute the loss

                loss.backward()

                yi, _ = model(x, keys=keys, train=False)

                input  = x[0].view(arg.size, 28, 28).data.cpu().numpy()
                target = t[0].view(arg.size, 28, 28).data.cpu().numpy()
                output_inf   = yi[0].view(arg.size, 28, 28).data.cpu().numpy()
                output_train = yt[0].view(arg.size, 28, 28).data.cpu().numpy()

                plt.figure(figsize=(arg.size*3, 4*3))
                for col in range(arg.size):

                    ax = plt.subplot(4, arg.size, col + 1)
                    ax.imshow(target[col], cmap='gray_r')
                    clean(ax)

                    if col == 0:
                        ax.set_ylabel('target')

                    ax = plt.subplot(4, arg.size, col + arg.size + 1)
                    ax.imshow(input[col], cmap='gray_r')
                    clean(ax)
                    ax.set_xlabel( '{:.2}, {:.2}'.format(keys[0, col], - keys.grad[0, col] ) )

                    if col == 0:
                        ax.set_ylabel('input')

                    ax = plt.subplot(4, arg.size, col + arg.size * 2 + 1)
                    ax.imshow(output_inf[col], cmap='gray_r')
                    clean(ax)

                    if col == 0:
                        ax.set_ylabel('inference')

                    ax = plt.subplot(4, arg.size, col + arg.size * 3 + 1)
                    ax.imshow(output_train[col], cmap='gray_r')
                    clean(ax)

                    if col == 0:
                        ax.set_ylabel('training')

                plt.savefig('./mnist-sort/{}/mnist.{:04}.pdf'.format(r, i))

                # plt.figure(figsize=(6, 2))
                # ax = plt.subplot(121)
                # ax.imshow(bottom.weight.data.view(28, 28), cmap='RdYlBu')
                # # ax.colorbar()
                # ax = plt.subplot(122)
                # ax.imshow(bottom.weight.grad.data.view(28, 28), cmap='RdYlBu')
                # # ax.title('{:.2}-{:.2}'.format(bottom.weight.grad.data.min(), bottom.weight.grad.data.max()))
                # plt.tight_layout()
                # plt.savefig('./mnist-sort/{}/weights.{:04}.pdf'.format(r, i))

                # sys.exit()

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
                        x, t, l = gen(arg.batch, test, arg.size)

                        if arg.cuda:
                            x, t, l = x.cuda(), t.cuda(), l.cuda()

                        x, t, l = Variable(x), Variable(t), Variable(l)

                        keys = tokeys(x.view(arg.batch * arg.size, 1, 28, 28))
                        keys = keys.view(arg.batch, arg.size)

                        # Sort the keys, and sort the labels, and see if the resulting indices match
                        _, gold = torch.sort(l, dim=1)
                        _, mine = torch.sort(keys, dim=1)

                        tot += x.size(0)
                        correct += ((gold != mine).sum(dim=1) == 0).sum().item()

                    print('acc', correct/tot)

                    results[r, i//arg.dot_every] = np.mean(correct/tot)

                    tbw.add_scalar('mnist-sort/testloss/{}/{}'.format(arg.size, r), correct/tot, i * arg.batch)

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

    parser.add_argument("-I", "--loss",
                        dest="loss",
                        help="Whether to backwards-sort the target to provide a loss at every step. (plain, means, separate)",
                        default='plain', type=str)


    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(options)
