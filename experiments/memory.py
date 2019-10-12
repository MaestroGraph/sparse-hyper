from _context import sparse

from sparse import util
from util import d

import torch

import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import nn
from torch.autograd import Variable
from tqdm import trange

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, os

"""
Memory layer experiment. Autoencoder with a small set of (learned) codes, arranged in 
an nD grid.

The encoder picks a single code in a sparse manner.

"""

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    [s.set_visible(False) for s in axes.spines.values()]
    axes.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)

class Model(nn.Module):

    def __init__(self, data_size, latent_size=(5, 5, 128), depth=3, gadditional=2, radditional=4, region=0.2,
                 method='clamp', sigma_scale=1.0, min_sigma=0.01):
        super().__init__()

        self.method, self.gadditional, self.radditional = method, gadditional, radditional
        self.sigma_scale, self.min_sigma = sigma_scale, min_sigma

        # latent space
        self.latent = nn.Parameter(torch.randn(size=latent_size))
        self.region = [int(r*region) for r in latent_size[:-1]]

        ln = len(latent_size)
        emb_size = latent_size[-1]

        c, h, w = data_size

        cs = [c] + [2**(d+4) for d in range(depth)]

        div = 2 ** depth

        modules = []

        for d in range(depth):
            modules += [
                nn.Conv2d(cs[d], cs[d+1], 3, padding=1), nn.ReLU(),
                nn.Conv2d(cs[d+1], cs[d+1], 3, padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 2))
            ]

        modules += [
            util.Flatten(),
            nn.Linear(cs[-1] * (h//div) * (w//div), 1024), nn.ReLU(),
            nn.Linear(1024, len(latent_size)) # encoder produces a cont. index tuple (ln -1 for the means, 1 for the sigma)
        ]

        self.encoder = nn.Sequential(*modules)

        upmode = 'bilinear'
        cl = lambda x : int(math.ceil(x))



        modules = [
            nn.Linear(emb_size, cs[-1] * cl(h/div) * cl(w/div)), nn.ReLU(),
            util.Reshape( (cs[-1], cl(h/div), cl(w/div)) )
        ]

        for d in range(depth, 0, -1):
            modules += [
                nn.Upsample(scale_factor=2, mode=upmode),
                nn.ConvTranspose2d(cs[d], cs[d], 3, padding=1), nn.ReLU(),
                nn.ConvTranspose2d(cs[d], cs[d-1], 3, padding=1), nn.ReLU()
            ]

        modules += [
            nn.ConvTranspose2d(c, c,  (3, 3), padding=1), nn.Sigmoid(),
            util.Lambda(lambda x : x[:, :, :h, :w]) # crop out any extra pixels due to rounding errors
        ]
        self.decoder = nn.Sequential(*modules)

        self.smp = True

    def sample(self, smp):
        self.smp = smp

    def forward(self, x):

        b, c, h, w = x.size()

        params = self.encoder(x)
        ls = self.latent.size()
        s, e = ls[:-1], ls[-1]

        assert params.size() == (b, len(ls))

        means  = sparse.transform_means(params[:, None, None,  :-1], s, method=self.method)
        sigmas = sparse.transform_sigmas(params[:, None, None, -1], s, min_sigma=self.min_sigma) * self.sigma_scale

        if self.smp:

            indices = sparse.ngenerate(means, self.gadditional, self.radditional, rng=s, relative_range=self.region, cuda=x.is_cuda)
            vs = (2**len(s) + self.radditional + self.gadditional)

            assert indices.size() == (b, 1, vs, len(s)), f'{indices.size()}, {(b, 1, vs, len(s))}'
            indfl = indices.float()

            # Mask for duplicate indices
            dups = util.nduplicates(indices).to(torch.bool)

            # compute (unnormalized) densities under the given MVNs (proportions)
            props = sparse.densities(indfl, means, sigmas).clone()
            assert props.size() == (b, 1, vs, 1) #?

            props[dups, :] = 0
            props = props / props.sum(dim=2, keepdim=True)  # normalize over all points of a given index tuple

            weights = props.sum(dim=-1)  # - sum out the MVNs

            assert indices.size() == (b, 1, vs, len(s))
            assert weights.size() == (b, 1, vs)

            indices, weights = indices.squeeze(1), weights.squeeze(1)

        else:
            vs = 1
            indices = means.floor().to(torch.long).detach().squeeze(1)

        # Select a single code from the latent space (per instance in batch).
        # When sampling, this is a weighted sum, when not sampling, just one.
        indices = indices.view(b*vs, len(s))

        # checks to prevent segfaults
        if util.contains_nan(indices):

            print(params)
            raise Exception('Indices contain NaN')

        if indices[:, 0].max() >= s[0] or indices[:, 1].max() >= s[1]:

            print(indices.max())
            print(params)
            raise Exception('Indices out of bounds')

        if len(s) == 1:
            code = self.latent[indices[:, 0], :]
        elif len(s) == 2:
            code = self.latent[indices[:, 0], indices[:, 1], :]
        elif len(s) == 3:
            code = self.latent[indices[:, 0], indices[:, 1], indices[:, 2], :]
        else:
            raise Exception(f'Dimensionality above 3 not supported.')
            # - ugly hack, until I figure out how to do this for n dimensions

        assert code.size() == (b*vs, e), f'{code.size()} --- {(b*vs, e)}'

        if self.smp:
            code = code.view(b, vs, e)
            code = code * weights[:, :, None]
            code = code.sum(dim=1)
        else:
            code = code.view(b, e)

        assert code.size() == (b, e)

        # Decode
        result = self.decoder(code)

        assert result.size() == (b, c, h, w), f'{result.size()} --- {(b, c, h, w)}'

        return result

def go(arg):

    util.makedirs('./memory/')

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir)
    tfms = transforms.Compose([transforms.ToTensor()])

    if (arg.task == 'mnist'):

        shape = (1, 28, 28)
        num_classes = 10

        data = arg.data + os.sep + arg.task

        if arg.final:
            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=tfms)
            trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch_size, shuffle=True, num_workers=0)

            test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=ToTensor())
            testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch_size, shuffle=False, num_workers=0)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=tfms)

            trainloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

    elif (arg.task == 'cifar10'):

        shape = (3, 32, 32)
        num_classes = 10

        data = arg.data + os.sep + arg.task

        if arg.final:
            train = torchvision.datasets.CIFAR10(root=data, train=True, download=True, transform=tfms)
            trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch, shuffle=True, num_workers=2)
            test = torchvision.datasets.CIFAR10(root=data, train=False, download=True, transform=ToTensor())
            testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.CIFAR10(root=data, train=True, download=True, transform=tfms)

            trainloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=arg.batch,
                                    sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

    elif arg.task == 'ffhq':

        transform = ToTensor()
        shape = (3, 128, 128)

        trainset = torchvision.datasets.ImageFolder(root=arg.data+os.sep+'train',
                                                    transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=arg.batch,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=arg.data+os.sep+'valid',
                                                   transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=arg.batch,
                                                 shuffle=False, num_workers=2)

    else:
        raise Exception('Task {} not recognized'.format(arg.task))

    model = Model(
        data_size=shape, latent_size=arg.lgrid, gadditional=arg.gadditional,
        radditional=arg.radditional, region=arg.region, method=arg.edges,
        sigma_scale=arg.sigma_scale, min_sigma=arg.min_sigma)

    if arg.cuda:
        model.cuda()

    opt = torch.optim.Adam(params=model.parameters(), lr=arg.lr)

    seen = 0
    for e in range(arg.epochs):
        print('epoch', e)

        model.train(True)

        for i, (inputs, _) in enumerate(tqdm.tqdm(trainloader, 0)):

            if arg.limit is not None and i > arg.limit:
                break

            b, c, h, w = inputs.size()
            seen += b

            model.sample(random.random() < arg.sample_prob) # use sampling only on some proportion of batches

            if arg.cuda:
                inputs = inputs.cuda()

            inputs = Variable(inputs)

            opt.zero_grad()

            outputs = model(inputs)

            loss = F.binary_cross_entropy(outputs, inputs.detach())

            loss.backward()

            opt.step()

            tbw.add_scalar('memory/loss', loss.item()/b, seen)

        if e % arg.plot_every == 0 and len(arg.lgrid) == 3:
            with torch.no_grad():

                codes = model.latent.data.view(-1, arg.lgrid[-1])
                images = model.decoder(codes)

                h, w = arg.lgrid[:2]

                plt.figure(figsize=(w, h))

                s = 1
                for i in range(h):
                    for j in range(w):

                        ax = plt.subplot(h, w, s)
                        ax.imshow(images[s-1].permute(1, 2, 0).squeeze().cpu(), cmap='Greys_r')

                        clean(ax)

                        s += 1

                plt.savefig(f'memory/latent.{e:03}.pdf')


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs",
                        default=250, type=int)

    parser.add_argument("-b", "--batch",
                        dest="batch",
                        help="Batch size",
                        default=64, type=int)

    parser.add_argument("-d", "--depth",
                        dest="depth",
                        help="Depth",
                        default=3, type=int)

    parser.add_argument("--task",
                        dest="task",
                        help="Dataset to model (mnist, cifar10)",
                        default='mnist', type=str)

    parser.add_argument("--latent-grid",
                        dest="lgrid",
                        help="Dimensionality of the latent codes. The last dimension represents the latent vector dimension.",
                        nargs='+',
                        default=[25, 25, 64], type=int)

    parser.add_argument("-a", "--gadditional",
                        dest="gadditional",
                        help="Number of additional points sampled globally per index-tuple",
                        default=2, type=int)

    parser.add_argument("-A", "--radditional",
                        dest="radditional",
                        help="Number of additional points sampled locally per index-tuple",
                        default=4, type=int)

    parser.add_argument("-R", "--range",
                        dest="region",
                        help="Range in which the local points are sampled (as a proportion of the whole space)",
                        default=0.2, type=float)

    parser.add_argument("-p", "--plot-every",
                        dest="plot_every",
                        help="Numer of epochs to wait between plotting",
                        default=1, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("--limit",
                        dest="limit",
                        help="Limit.",
                        default=None, type=int)

    parser.add_argument("-r", "--seed",
                        dest="seed",
                        help="Random seed",
                        default=0, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data')

    parser.add_argument("--sample-prob",
                        dest="sample_prob",
                        help="Sample probability (with this probability we sample index tuples).",
                        default=0.5, type=float)

    parser.add_argument("--edges", dest="edges",
                        help="Which operator to use to fit continuous index tuples to the required range.",
                        default='clamp', type=str)

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Minimum value of sigma.",
                        default=0.01, type=float)

    parser.add_argument("-S", "--sigma-scale",
                        dest="sigma_scale",
                        help="Scalar applied to sigmas.",
                        default=0.5, type=float)

    args = parser.parse_args()

    print('OPTIONS', args)

    go(args)
