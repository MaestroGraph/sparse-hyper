from _context import sparse

import torch, torchvision
import numpy as np

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch.autograd import Variable

from torch import nn
import torch.nn.functional as F

from argparse import ArgumentParser

import os, sys, math

from sparse import util, NASLayer, Convolution

from tqdm import trange, tqdm

from tensorboardX import SummaryWriter

"""
This experiment trains a simple, fully connected three-layer MLP, following the baseline from Louizos 2018.
We aim to show that in the very low density regime, the sparse layer is a competitive approach.

The tasks are simple classification on mnist, cifar10 and cifar100.

TODO: Test temp version.

"""

BATCH_SIZES    = [128]
LEARNING_RATES = [0.00005, 0.0001, 0.0005, 0.001, 0.005]

def getrng(p, size):
    return [max(1, int(math.floor(p * s))) for s in size]

def getmodel(arg, insize, numcls):

    h1, h2 = arg.hidden

    # if arg.method == 'l1' or arg.method == 'lp':
    #
    #     one = nn.Linear(util.prod(insize), h1)
    #     two = nn.Linear(h1, h2)
    #     three = nn.Linear(h2, numcls)
    #
    #     model = nn.Sequential(
    #         util.Flatten(),
    #         one, nn.Sigmoid(),
    #         two, nn.Sigmoid(),
    #         three, nn.Softmax()
    #     )

    if arg.method == 'nas':
        """
        Non-templated NAS model
        """

        rng = getrng(arg.range[0], (h1, ) + insize)

        c = arg.k[0]

        one = NASLayer(
            in_size=insize, out_size=(h1,), k=h1*c,
            gadditional=arg.gadditional[0], radditional=arg.radditional[0], region=rng, has_bias=True,
            fix_values=arg.fix_values,
            min_sigma=arg.min_sigma,
            template=None,
            learn_cols=None,
            chunk_size=c
        )

        rng = getrng(arg.range[1], (h2, h1))
        c = arg.k[1]

        two = NASLayer(
            in_size=(h1,), out_size=(h2,), k=h2*c,
            gadditional=arg.gadditional[1], radditional=arg.radditional[1], region=rng, has_bias=True,
            fix_values=arg.fix_values,
            min_sigma=arg.min_sigma,
            template=None,
            learn_cols=None,
            chunk_size=c
        )

        rng = getrng(arg.range[2], (numcls, h2))
        c = arg.k[2]

        three = NASLayer(
            in_size=(h2,), out_size=(numcls,), k=numcls*c,
            gadditional=arg.gadditional[2], radditional=arg.radditional[2], region=rng, has_bias=True,
            fix_values=arg.fix_values,
            min_sigma=arg.min_sigma,
            template=None,
            learn_cols=None,
            chunk_size=c
        )

        model = nn.Sequential(
            one, nn.Sigmoid(),
            two, nn.Sigmoid(),
            three, nn.Softmax(),
        )

    elif arg.method == 'nas-temp':
        """
        Templated NAS model. Fixed output dimensions.
        """

        rng = getrng(arg.range[0], (insize[1], insize[2]))
        c = arg.k[0]

        template = torch.arange(h1, dtype=torch.long)[:, None].expand(h1, c).contiguous().view(h1*c, 1)
        template = torch.cat([template, torch.zeros(h1*c, 3, dtype=torch.long)], dim=1)

        one = NASLayer(
            in_size=insize, out_size=(h1,), k=h1*c,
            gadditional=arg.gadditional[0], radditional=arg.radditional[0], region=rng, has_bias=True,
            fix_values=arg.fix_values,
            min_sigma=arg.min_sigma,
            template=template,
            learn_cols=(1, 2, 3) if insize[0] > 1 else (2, 3),
            chunk_size=c
        )

        rng = getrng(arg.range[1], (h1, ))
        c = arg.k[1]

        template = torch.arange(h2, dtype=torch.long)[:, None].expand(h2, c).contiguous().view(h2 * c, 1)
        template = torch.cat([template, torch.zeros(h2*c, 1, dtype=torch.long)], dim=1)

        two = NASLayer(
            in_size=(h1,), out_size=(h2,), k=h2*c,
            gadditional=arg.gadditional[1], radditional=arg.radditional[1], region=rng, has_bias=True,
            fix_values=arg.fix_values,
            min_sigma=arg.min_sigma,
            template=template,
            learn_cols=(1,),
            chunk_size=c
        )

        rng = getrng(arg.range[2], (h2, ))
        c = arg.k[2]

        template = torch.arange(numcls, dtype=torch.long)[:, None].expand(numcls, c).contiguous().view(numcls * c, 1)
        template = torch.cat([template, torch.zeros(numcls*c, 1, dtype=torch.long)], dim=1)

        three = NASLayer(
            in_size=(h2,), out_size=(numcls,), k=numcls*c,
            gadditional=arg.gadditional[2], radditional=arg.radditional[2], region=rng, has_bias=True,
            fix_values=arg.fix_values,
            min_sigma=arg.min_sigma,
            template=template,
            learn_cols=(1,),
            chunk_size=c
        )

        model = nn.Sequential(
            one, nn.Sigmoid(),
            two, nn.Sigmoid(),
            three, nn.Softmax(),
        )
    elif arg.method == 'nas-conv':
        """
        Convolutional NAS model.
        """
        c1, c2 = h1, h2

        one = Convolution(in_size=(1, 28, 28), out_channels=c1, k=arg.k[0], kernel_size=7,
                          gadditional=arg.gadditional[0], radditional=arg.radditional[1], rprop=arg.range[0],
                          fix_values=arg.fix_values, has_bias=True)

        two = Convolution(in_size=(c1, 14, 14), out_channels=c2, k=arg.k[1], kernel_size=7,
                          gadditional=arg.gadditional[1], radditional=arg.radditional[1], rprop=arg.range[1],
                          fix_values=arg.fix_values, has_bias=True)

        three = Convolution(in_size=(c2, 7, 7), out_channels=numcls, k=arg.k[2], kernel_size=7,
                          gadditional=arg.gadditional[2], radditional=arg.radditional[2], rprop=arg.range[2],
                          fix_values=arg.fix_values, has_bias=True)

        model = nn.Sequential(
            one, nn.Sigmoid(), nn.MaxPool2d(2),
            two, nn.Sigmoid(), nn.MaxPool2d(2),
            three, nn.Sigmoid(), nn.MaxPool2d(2),
            util.Lambda(lambda x : x.mean(dim=-1).mean(dim=-1)), # global average pool
            nn.Softmax()
        )

    elif arg.method == 'conv':
        c1, c2 = h1, h2

        one =  nn.Conv2d(insize[0], c1, kernel_size=3, padding=1)
        two = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        three = nn.Conv2d(c2, numcls, kernel_size=3, padding=1)

        model = nn.Sequential(
            one, nn.Sigmoid(), nn.MaxPool2d(2),
            two, nn.Sigmoid(), nn.MaxPool2d(2),
            three, nn.Sigmoid(), nn.MaxPool2d(2),
            util.Lambda(lambda x : x.mean(dim=-1).mean(dim=-1)), # global average pool
            nn.Softmax()
        )

    elif arg.method == 'one':
        """
        Convolutional NAS model.
        """
        c1, c2 = h1, h2

        one = Convolution(in_size=(1, 28, 28), out_channels=c1, k=arg.k[0], kernel_size=7,
                          gadditional=arg.gadditional[0], radditional=arg.radditional[1], rprop=arg.range[0],
                          fix_values=arg.fix_values, has_bias=True)

        two = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        three = nn.Conv2d(c2, numcls, kernel_size=3, padding=1)

        model = nn.Sequential(
            one, nn.Sigmoid(), nn.MaxPool2d(2),
            two, nn.Sigmoid(), nn.MaxPool2d(2),
            three, nn.Sigmoid(), nn.MaxPool2d(2),
            util.Lambda(lambda x : x.mean(dim=-1).mean(dim=-1)), # global average pool
            nn.Softmax()
        )
    elif arg.method == 'two':
        """
        Convolutional NAS model.
        """
        c1, c2 = h1, h2

        one = Convolution(in_size=(1, 28, 28), out_channels=c1, k=arg.k[0], kernel_size=7,
                          gadditional=arg.gadditional[0], radditional=arg.radditional[1], rprop=arg.range[0],
                          fix_values=arg.fix_values, has_bias=True)

        two = Convolution(in_size=(c1, 14, 14), out_channels=c2, k=arg.k[1], kernel_size=7,
                          gadditional=arg.gadditional[1], radditional=arg.radditional[1], rprop=arg.range[1],
                          fix_values=arg.fix_values, has_bias=True)

        three = nn.Conv2d(c2, numcls, kernel_size=3, padding=1)

        model = nn.Sequential(
            one, nn.Sigmoid(), nn.MaxPool2d(2),
            two, nn.Sigmoid(), nn.MaxPool2d(2),
            three, nn.Sigmoid(), nn.MaxPool2d(2),
            util.Lambda(lambda x : x.mean(dim=-1).mean(dim=-1)), # global average pool
            nn.Softmax()
        )

    else:
        raise Exception('Method {} not recognized'.format(arg.method))

    if arg.cuda:
        model.cuda()

    return model, one, two, three

def single(arg):

    tbw = SummaryWriter()
    #
    # lambd = torch.logspace(arg.rfrom, arg.rto, arg.rnum)[arg.control].item()
    #
    # print('lambda ', lambd)

    # Grid search over batch size/learning rate
    # -- Set up model

    insize = (1, 28, 28) if arg.task == 'mnist' else (3, 32, 32)
    numcls = 100 if arg.task == 'cifar100' else 10

    # Repeat runs with chosen hyperparameters
    accuracies = []
    densities = []

    for _ in trange(arg.repeats):

        if arg.task == 'mnist':
            if arg.final:
                data = arg.data + os.sep + arg.task

                train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=ToTensor())
                trainloader = torch.utils.data.DataLoader(train, batch_size=arg.bs, shuffle=True, num_workers=2)

                test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=ToTensor())
                testloader = torch.utils.data.DataLoader(test, batch_size=arg.bs, shuffle=False, num_workers=2)
            else:

                NUM_TRAIN = 45000
                NUM_VAL = 5000
                total = NUM_TRAIN + NUM_VAL

                train = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=ToTensor())

                trainloader = DataLoader(train, batch_size=arg.bs, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
                testloader = DataLoader(train, batch_size=arg.bs,
                                        sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

        elif (arg.task == 'cifar10'):

            data = arg.data + os.sep + arg.task

            if arg.final:
                train = torchvision.datasets.CIFAR10(root=data, train=True, download=True, transform=ToTensor())
                trainloader = torch.utils.data.DataLoader(train, batch_size=arg.bs, shuffle=True, num_workers=2)
                test = torchvision.datasets.CIFAR10(root=data, train=False, download=True, transform=ToTensor())
                testloader = torch.utils.data.DataLoader(test, batch_size=arg.bs, shuffle=False, num_workers=2)

            else:
                NUM_TRAIN = 45000
                NUM_VAL = 5000
                total = NUM_TRAIN + NUM_VAL

                train = torchvision.datasets.CIFAR10(root=data, train=True, download=True, transform=ToTensor())

                trainloader = DataLoader(train, batch_size=arg.bs, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
                testloader = DataLoader(train, batch_size=arg.bs,
                                        sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

        else:
            raise Exception('Task {} not recognized'.format(arg.task))

        model, one, two, three = getmodel(arg, insize, numcls) # new model
        opt = torch.optim.Adam(model.parameters(), lr=arg.lr)

        # Train for fixed number of epochs
        i = 0
        for e in range(arg.epochs):

            model.train(True)
            for input, labels in tqdm(trainloader):
                opt.zero_grad()

                if arg.cuda:
                    input, labels = input.cuda(), labels.cuda()
                input, labels = Variable(input), Variable(labels)

                output = model(input)

                loss = F.cross_entropy(output, labels)
                loss.backward()

                tbw.add_scalar('sparsity/loss', loss.data.item(), i * arg.bs)
                i += 1

                opt.step()

            # Compute accuracy on test set
            with torch.no_grad():
                model.train(False)

                total, correct = 0.0, 0.0
                for input, labels in testloader:
                        opt.zero_grad()

                        if arg.cuda:
                            input, labels = input.cuda(), labels.cuda()
                        input, labels = Variable(input), Variable(labels)

                        output = model(input)

                        outcls = output.argmax(dim=1)

                        total   += outcls.size(0)
                        correct += (outcls == labels).sum().item()

                acc = correct / float(total)

                print('\nepoch {}: {}\n'.format(e, acc))
                tbw.add_scalar('sparsity/test acc', acc, e)

    #     # Compute density
    #     total = util.prod(insize) * arg.hidden
    #
    #     kt = arg.control[] * (arg.hidden[0] +  + numcls)
    #
    #     if arg.method == 'l1' or arg.method == 'lp':
    #         density = (one.weight > 0.0001).sum().item() / float(total)
    #     elif arg.method == 'nas' or arg.method == 'nas-temp':
    #         density = kt / total
    #     else:
    #         raise Exception('Method {} not recognized'.format(arg.method))
    #
    #     accuracies.append(acc)
    #     densities.append(density)
    #
    # print('accuracies: ', accuracies)
    # print('densities: ', densities)
    #
    # if arg.method == 'lp':
    #     if arg.p == 0.2:
    #         name = 'l5'
    #     elif arg.p == 0.5:
    #         name = 'l2'
    #     elif arg.p == 1.0:
    #         name = 'l1'
    #     else:
    #         name = 'l' + arg.p
    # else:
    #     name = arg.method
    #
    # # Save to CSV
    # np.savetxt(
    #     'results.{}.{}.csv'.format(name, arg.control),
    #     torch.cat([
    #             torch.tensor(accuracies, dtype=torch.float)[:, None],
    #             torch.tensor(densities, dtype=torch.float)[:, None]
    #         ], dim=1).numpy(),
    # )

    print('Finished')

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-H", "--hidden",
                        dest="hidden",
                        nargs=2,
                        help="Sizes of the two hidden layers",
                        default=[300, 100],
                        type=int)

    parser.add_argument("-k", "--points-per-out",
                        dest="k",
                        nargs=3,
                        help="Number of sparse points for each output node.",
                        default=[1, 1, 1], type=int)

    parser.add_argument("-l", "--lr",
                        dest="lr",
                        help="Learning rate (ignored in sweep)",
                        default=0.001, type=float)

    parser.add_argument("-b", "--batch ",
                        dest="bs",
                        help="Batch size (ignored in sweep)",
                        default=64, type=int)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs",
                        default=50, type=int)

    parser.add_argument("-m", "--method",
                        dest="method",
                        help="Method to use (lp, nas) ",
                        default='nas-temp', type=str)

    parser.add_argument("-P", "--lp-p",
                        dest="p",
                        help="Exponent in lp reg",
                        default=2.0, type=float)

    parser.add_argument("-t", "--task",
                        dest="task",
                        help="Task to use (mnist, cifar10, cifar100) ",
                        default='mnist', type=str)

    parser.add_argument("-a", "--gadditional",
                        dest="gadditional",
                        nargs=3,
                        help="Number of additional points sampled globally per index-tuple (NAS)",
                        default=[32, 6, 2], type=int)

    parser.add_argument("-A", "--radditional",
                        dest="radditional",
                        nargs=3,
                        help="Number of additional points sampled locally per index-tuple (NAS)",
                        default=[32, 6, 2], type=int)

    parser.add_argument("-R", "--range",
                        dest="range",
                        nargs=3,
                        help="Range in which the local points are sampled (NAS)",
                        default=[0.3, 0.2, 0.2], type=float)

    parser.add_argument("-r", "--repeats",
                        dest="repeats",
                        help="Number of times to repeat the final experiment (once the hyperparameters are chosen).",
                        default=10, type=int)

    parser.add_argument("--seed",
                        dest="seed",
                        help="Random seed",
                        default=4, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data')

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Minimal sigma value",
                        default=0.01, type=float)

    parser.add_argument("--rfrom",
                        dest="rfrom",
                        help="Minimal control value (for lp baselines)",
                        default=0.00001, type=float)

    parser.add_argument("--rto",
                        dest="rto",
                        help="Maximal control value (for lp baselines)",
                        default=1.0, type=float)

    parser.add_argument("--rnum",
                        dest="rnum",
                        help="Number of control parameters (for lp baseline)",
                        default=10, type=int)

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set.",
                        action="store_true")

    parser.add_argument("-F", "--fix-values", dest="fix_values",
                        help="Whether to fix all values to 1 in the NAS model.",
                        action="store_true")

    args = parser.parse_args()

    print('OPTIONS', args)

    single(args)
