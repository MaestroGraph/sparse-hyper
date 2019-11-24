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

import torch.distributions as ds

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, os

"""
Experiment to test bias of gradient estimator. Simple encoder/decoder with discrete latent space.

"""


def sample_gumbel(shape, eps=1e-20, cuda=False):
    U = torch.rand(shape, device=d(cuda))
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbelize(logits, temperature=1.0):
    y = logits + sample_gumbel(logits.size(), cuda=logits.is_cuda)
    return y / temperature

def gradient(models):
    """
    Returns the gradient of the given models as a single vector
    :param models:
    :return:
    """
    gs = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                gs.append(param.grad.data.view(-1))

    return torch.cat(gs, dim=0)


def num_params(models):
    """
    Returns the gradient of the given models as a single vector
    :param models:
    :return:
    """
    gs = 0
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                gs += param.view(-1).size(0)

    return gs

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    [s.set_visible(False) for s in axes.spines.values()]
    axes.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)

class Encoder(nn.Module):

    def __init__(self, data_size, latent_size=128, depth=3):
        super().__init__()

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
            nn.Linear(1024, latent_size) # encoder produces a cont. index tuple (ln -1 for the means, 1 for the sigma)
        ]

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):

        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, data_size, latent_size=128, depth=3):
        super().__init__()

        upmode = 'bilinear'

        c, h, w = data_size
        cs = [c] + [2**(d+4) for d in range(depth)]

        div = 2 ** depth
        cl = lambda x : int(math.ceil(x))

        modules = [
            nn.Linear(latent_size, cs[-1] * cl(h/div) * cl(w/div)), nn.ReLU(),
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

    def forward(self, x):
        return self.decoder(x)

def go(arg):

    util.makedirs('./bias/')

    if not os.path.exists('./bias/cached.npz'):

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

        encoder = Encoder(shape, latent_size=arg.latent_size, depth=arg.depth)
        decoder = Decoder(shape, latent_size=arg.latent_size, depth=arg.depth)

        if arg.cuda:
            encoder.cuda()
            decoder.cuda()

        opt = torch.optim.Adam(params=list(encoder.parameters()) + list(decoder.parameters()), lr=arg.lr)

        nparms = num_params([encoder])
        print(f'{nparms} parameters in encoder.')

        seen = 0
        l = arg.latent_size
        ti = random.sample(range(nparms), arg.num_params) # random indices of parameters for which to test the gradient
        k = arg.k

        # Train for a fixed nr of instances (with the true gradient)
        for e in range(arg.epochs):
            print('epoch', e)

            for i, (inputs, _) in enumerate(trainloader):

                b, c, h, w = inputs.size()

                if arg.cuda:
                    inputs = inputs.cuda()

                # compute actual gradient
                opt.zero_grad()

                latent = encoder(inputs)
                latent = F.softmax(latent, dim=1)

                dinp = torch.eye(l, device=d(arg.cuda))[None, :, :].expand(b, l, l).reshape(b*l, l)
                dout = decoder(dinp)

                assert dout.size() == (b*l, c, h, w)

                target = inputs.detach()[:, None, :, :, :].expand(b, l, c, h, w).reshape(b*l, c, h, w)

                loss = F.binary_cross_entropy(dout, target, reduction='none')
                loss = loss.sum(dim=1).sum(dim=1).sum(dim=1).view(b, l)

                loss = (loss * latent).sum(dim=1).mean()

                loss.backward()

                true_gradient = gradient([encoder, decoder])
                true_gradient = true_gradient[ti]

                opt.step()

        inputs, _ = next(iter(trainloader))
        if arg.cuda:
            inputs = inputs.cuda()

        b, c, h, w = inputs.size()

        # compute true gradient
        opt.zero_grad()

        latent = encoder(inputs)
        latent = F.softmax(latent, dim=1)

        dinp = torch.eye(l, device=d(arg.cuda))[None, :, :].expand(b, l, l).reshape(b*l, l)
        dout = decoder(dinp)

        assert dout.size() == (b*l, c, h, w)

        target = inputs.detach()[:, None, :, :, :].expand(b, l, c, h, w).reshape(b*l, c, h, w)

        loss = F.binary_cross_entropy(dout, target, reduction='none')
        loss = loss.sum(dim=1).sum(dim=1).sum(dim=1).view(b, l)

        loss = (loss * latent).sum(dim=1).mean()

        loss.backward()

        true_gradient = gradient([encoder])
        true_gradient = true_gradient[ti]

        # - Estimate the bias for the uninformed sampler

        uste = torch.zeros((arg.samples, len(ti),), device=d(arg.cuda))

        # Unbiased, uninformed STE
        for s in trange(arg.samples):
            opt.zero_grad()

            ks = [random.sample(range(arg.latent_size), k) for _ in range(b)]
            ks = torch.tensor(ks, device=d(arg.cuda))

            latent = encoder(inputs)
            latent = torch.gather(latent, dim=1, index=ks); assert latent.size() == (b, k)
            latent = F.softmax(latent, dim=1)

            dinp = torch.zeros(size=(b*k, l), device=d(arg.cuda))
            dinp.scatter_(dim=1, index=ks.view(b*k, 1), value=1)
            dout = decoder(dinp)

            assert dout.size() == (b * k, c, h, w)

            target = inputs.detach()[:, None, :, :, :].expand(b, k, c, h, w).reshape(b * k, c, h, w)

            loss = F.binary_cross_entropy(dout, target, reduction='none')
            loss = loss.sum(dim=1).sum(dim=1).sum(dim=1).view(b, k)

            loss = (loss * latent).sum(dim=1).mean()

            loss.backward()

            samp_gradient = gradient([encoder])
            uste[s, :] = samp_gradient[ti]

            del loss

        iste = torch.zeros((arg.samples, len(ti),), device=d(arg.cuda))

        # Unbiased, informed STE
        # This behaves like the USTE, but ensures that the argmax is always included in the sample
        for s in trange(arg.samples):
            opt.zero_grad()

            latent = encoder(inputs)

            ks = [random.sample(range(arg.latent_size-1), k-1) for _ in range(b)]
            ks = torch.tensor(ks, device=d(arg.cuda))
            am = latent.argmax(dim=1, keepdim=True)
            ks[ks > am] += 1

            ks = torch.cat([am, ks], dim=1)

            latent = torch.gather(latent, dim=1, index=ks); assert latent.size() == (b, k)
            latent = F.softmax(latent, dim=1)

            dinp = torch.zeros(size=(b * k, l), device=d())
            dinp.scatter_(dim=1, index=ks.view(b * k, 1), value=1)
            dout = decoder(dinp)

            assert dout.size() == (b * k, c, h, w)

            target = inputs.detach()[:, None, :, :, :].expand(b, k, c, h, w).reshape(b * k, c, h, w)

            loss = F.binary_cross_entropy(dout, target, reduction='none')
            loss = loss.sum(dim=1).sum(dim=1).sum(dim=1).view(b, k)

            loss = (loss * latent).sum(dim=1).mean()

            loss.backward()

            samp_gradient = gradient([encoder])
            iste[s, :] = samp_gradient[ti]

            del loss

        # Biased (?) gumbel STE
        # STE with gumbel noise

        gste = torch.zeros((arg.samples, len(ti),), device=d(arg.cuda))

        for s in trange(arg.samples):
            for _ in range(k):
                opt.zero_grad()

                latent = encoder(inputs)

                gumbelize(latent, temperature=arg.gumbel)
                latent = F.softmax(latent, dim=1)

                ks = latent.argmax(dim=1, keepdim=True)

                dinp = torch.zeros(size=(b, l), device=d())
                dinp.scatter_(dim=1, index=ks, value=1)

                dinp = (dinp - latent).detach() + latent # straight-through trick
                dout = decoder(dinp)

                assert dout.size() == (b, c, h, w)

                target = inputs.detach()

                loss = F.binary_cross_entropy(dout, target, reduction='none')
                loss = loss.sum(dim=1).sum(dim=1).sum(dim=1).view(b)
                loss = loss.mean()

                loss.backward()

                samp_gradient = gradient([encoder])
                gste[s, :] += samp_gradient[ti]

                del loss

            gste[s, :] /= k

        # Classical STE
        # cste = torch.zeros((arg.samples, len(ti),), device=d(arg.cuda))
        #
        # for s in trange(arg.samples):
        #     opt.zero_grad()
        #
        #     latent = encoder(inputs)
        #
        #     # gumbelize(latent, temperature=arg.gumbel)
        #     dist = ds.Categorical(logits=latent)
        #     ks = dist.sample()[:, None]
        #
        #     dinp = torch.zeros(size=(b, l), device=d())
        #     dinp.scatter_(dim=1, index=ks, value=1)
        #
        #     dinp = (dinp - latent).detach() + latent # straight-through trick
        #     dout = decoder(dinp)
        #
        #     assert dout.size() == (b, c, h, w)
        #
        #     target = inputs.detach()
        #
        #     loss = F.binary_cross_entropy(dout, target, reduction='none')
        #     loss = loss.sum(dim=1).sum(dim=1).sum(dim=1).view(b)
        #     loss = loss.mean()
        #
        #     loss.backward()
        #
        #     samp_gradient = gradient([encoder])
        #     cste[s, :] = samp_gradient[ti]
        #
        #     del loss

        uste = uste.cpu().numpy()
        iste = iste.cpu().numpy()
        gste = gste.cpu().numpy()
        tgrd = true_gradient.cpu().numpy()

        np.savez_compressed('./bias/cached.npz', uste=uste, iste=iste, gste=gste, tgrd=tgrd)

    else:
        res = np.load('./bias/cached.npz')
        uste, iste, gste, tgrd = res['uste'], res['iste'], res['gste'], res['tgrd']

    ind = tgrd != 0.0
    print(tgrd.shape, ind)

    print(f'{ind.sum()} derivatives out of {ind.shape} not equal to zero.')

    for nth, i in enumerate( np.arange(ind.shape[0])[ind][:40] ):

        plt.gcf().clear()

        unump = uste[:, i]
        inump = iste[:, i]
        gnump = gste[:, i]
        # cnump = cste[:, i].cpu().numpy()

        ulab = f'uninformed, var={unump.var():.4}'
        ilab = f'informed, var={inump.var():.4}'
        glab = f'Gumbel STE (t={arg.gumbel}) var={gnump.var():.4}'
        # clab = f'Classical STE var={cnump.var():.4}'

        plt.hist([unump, inump, gnump], color=['r', 'g', 'b'], label=[ulab, ilab, glab],bins='sturges')

        plt.axvline(x=unump.mean(), color='r')
        plt.axvline(x=inump.mean(), color='g')
        plt.axvline(x=gnump.mean(), color='b')
        # plt.axvline(x=cnump.mean(), color='c')
        plt.axvline(x=tgrd[i], color='k', label='true gradient')

        plt.title(f'estimates for parameter ... ({uste.shape[0]} samples)')

        plt.legend()
        util.basic()

        plt.savefig(f'./bias/histogram.{nth}.pdf')


    plt.gcf().clear()

    unump = uste[:, ind].mean(axis=0)
    inump = iste[:, ind].mean(axis=0)
    gnump = gste[:, ind].mean(axis=0)

    tnump = tgrd[ind]

    unump = np.abs(unump - tnump)
    inump = np.abs(inump - tnump)
    gnump = np.abs(gnump - tnump)

    ulab = f'uninformed, var={unump.var():.4}'
    ilab = f'informed, var={inump.var():.4}'
    glab = f'gumbel STE (t={arg.gumbel}) var={gnump.var():.4}'
    # clab = f'Classical STE var={cnump.var():.4}'

    plt.hist([unump, inump, gnump], color=['r', 'g', 'b'], label=[ulab, ilab, glab],bins='sturges')

    plt.axvline(x=unump.mean(), color='r')
    plt.axvline(x=inump.mean(), color='g')
    plt.axvline(x=gnump.mean(), color='b')
    # plt.axvline(x=cnump.mean(), color='c')

    plt.title(f'Absolute error between true gradient and estimate \n over {ind.sum()} parameters with nonzero gradient.')

    plt.legend()
    util.basic()

    if arg.range is not None:
        plt.xlim(*arg.range)

    plt.savefig(f'./bias/histogram.all.pdf')

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs to train (with the true gradient) before testing the estimator biases.",
                        default=5, type=int)

    parser.add_argument("-b", "--batch",
                        dest="batch",
                        help="Batch size",
                        default=8, type=int)

    parser.add_argument("-d", "--depth",
                        dest="depth",
                        help="Depth of the autoencoder (number of maxpooling operations).",
                        default=3, type=int)

    parser.add_argument("--num-params",
                        dest="num_params",
                        help="Depth",
                        default=50000, type=int)

    parser.add_argument("--task",
                        dest="task",
                        help="Dataset to model (mnist, cifar10)",
                        default='mnist', type=str)

    parser.add_argument("--latent-size",
                        dest="latent_size",
                        help="Size of the discrete latent space.",
                        default=128, type=int)

    parser.add_argument("--samples",
                        dest="samples",
                        help="Number of samples to take from the estimator.",
                        default=100, type=int)

    parser.add_argument("-p", "--plot-every",
                        dest="plot_every",
                        help="Number of epochs to wait between plotting",
                        default=1, type=int)

    parser.add_argument("-k", "--set-size",
                        dest="k",
                        help="Size of the sample (the set S). For the gumbel softmax, we average the estimate over k separate samples",
                        default=5, type=int)

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

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)

    parser.add_argument("-G", "--gumbel", dest="gumbel",
                        help="Gumbel temperature.",
                        default=1.0, type=float)

    parser.add_argument("--range", dest="range",
                        help="Range for the 'all' plot.",
                        nargs=2,
                        default=None, type=float)

    args = parser.parse_args()

    print('OPTIONS', args)

    go(args)
