from _context import sparse
from sparse import util

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

import random, tqdm, sys, math, os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from util import d

class MobBlock(nn.Module):
    """
    Inverted residual block (as in mobilenetv2)
    """
    def __init__(self,  c, wide=None, kernel=3):
        super().__init__()

        wide = 6 * c if wide is None else wide
        padding = int(math.floor(kernel/2))

        self.convs = nn.Sequential(
            nn.Conv2d(c, wide, kernel_size=1),
            #nn.BatchNorm2d(wide),
            nn.ReLU6(),
            nn.Conv2d(wide, wide, kernel_size=kernel, padding=padding, groups=wide),
            #nn.BatchNorm2d(wide),
            nn.ReLU6(),
            nn.Conv2d(wide, c, kernel_size=1),
            nn.BatchNorm2d(c), nn.ReLU6()
        )

    def forward(self, x):

        return self.convs(x) + x

class Convolution(nn.Module):
    """
    A "convolution" with a learned sparsity structure (as opposed to a fixed k x k grid)

    A sparse transformation is used to select k pixels from the input. Their channel
    vectors are concatenated and linearly transformed to the desired number of output
    channels (this is done for every output pixel).

    The pattern of input pixels, relative to the current output pixels is determined
    adaptively.
    """
    def __init__(self, in_size, out_size, k, gadditional, radditional, region, min_sigma=0.05, sigma_scale=0.05,
                 mmult=None, adaptive=False, modulo=True):
        """
        :param k: Number of connections to the input in total
        :param gadditional:
        :param radditional:
        :param region:
        """

        super().__init__()

        cin, hin, win = in_size
        cout, hout, wout = out_size

        assert hin > 2 and win > 2, 'Input resolution must be larger than 2x2 for the sparse convolution to work.'

        self.gadditional, self.radditional = gadditional, radditional

        self.region = (max(int(region*hin), 2), max(int(region*win), 2))

        self.in_size, self.out_size = in_size, out_size
        self.min_sigma, self.sigma_scale = min_sigma, sigma_scale
        self.adaptive = adaptive
        self.modulo = modulo

        self.mmult = (1.0 if modulo else 0.1) if mmult is None else mmult

        self.k = k
        self.unify = nn.Linear(k*cin, cout)

        # network that generates the coordinates and sigmas
        hidden = cin * 4
        self.toparams = nn.Sequential(
            nn.Linear(2+cin if adaptive else 2, hidden), nn.ReLU(),
            nn.Linear(hidden, k * 3) # two means, one sigma
        )

        self.register_buffer('mvalues', torch.ones((k,)))
        self.register_buffer('coords', util.coordinates((hin, win)))

        self.smp = True

        assert self.coords.size() == (2, hin, win)

    def sample(self, smp):
        self.smp = smp

    def hyper(self, x):

        assert x.size()[1:] == self.in_size
        b, c, h, w = x.size()
        k = self.k

        # the coordinates of the current pixels in parameters space
        # - the index tuples are described relative to these
        hw = torch.tensor((h, w), device=d(x), dtype=torch.float)
        mids = self.coords[None, :, :, :].expand(b, 2, h, w) * (hw - 1)[None, :, None, None]
        mids = mids.permute(0, 2, 3, 1)
        if not self.modulo:
            mids = util.inv(mids, mx=hw[None, None, None, :])
        mids = mids[:, :, :, None, :].expand(b, h, w, k, 2)

        # add coords to channels
        coords = self.coords[None, :, :, :].expand(b, 2, h, w)
        x = torch.cat([x, coords], dim=1) if self.adaptive else coords

        x = x.permute(0, 2, 3, 1)

        params = self.toparams(x)
        assert params.size() == (b, h, w, k * 3) # k index tuples per output pixel

        means  = params[:, :, :, :k*2].view(b, h, w, k, 2)
        sigmas = params[:, :, :, k*2:].view(b, h, w, k)
        values = self.mvalues[None, None, None, :].expand(b, h, w, k)

        means = mids + self.mmult * means

        s = (h, w)
        means  = sparse.transform_means(means, s, method='modulo' if self.modulo else 'sigmoid')
        sigmas = sparse.transform_sigmas(sigmas, s, min_sigma=self.min_sigma) * self.sigma_scale

        return means, sigmas, values

    def forward(self, x):

        assert x.size()[1:] == self.in_size

        b, c, h, w = x.size()
        k = self.k
        s = (h, w)

        means, sigmas, mvalues = self.hyper(x)

        # This is a bit confusing, but k is the chunk dimension here. This is because the sparse operation
        # only selects in the k separate input pixels, it doens not sum/merge them.
        # In other words, we add a separate tuple dimension.
        means   = means  [:, :, :, :, None, :]
        sigmas  = sigmas [:, :, :, :, None, :]
        mvalues = mvalues[:, :, :, :, None]

        if self.smp:
            # sample integer indices and values
            indices = sparse.ngenerate(means, self.gadditional, self.radditional, rng=s, relative_range=self.region, cuda=x.is_cuda)

            vs = (4 + self.radditional + self.gadditional)
            assert indices.size() == (b, h, w, k, vs, 2), f'{indices.size()}, {(b, h, w, k, vs, 2)}'

            indices = indices.view(b, h, w, k, vs, 2)
            indfl = indices.float()

            # Mask for duplicate indices
            dups = util.nduplicates(indices).to(torch.bool)

            # compute (unnormalized) densities under the given MVNs (proportions)
            props = sparse.densities(indfl, means, sigmas).clone() # (b, h, w, k, vs, 1)
            assert props.size() == (b, h, w, k, vs, 1)

            props[dups, :] = 0
            props = props / props.sum(dim=4, keepdim=True)  # normalize over all points of a given index tuple

            # weight the values by the proportions
            weights = mvalues[:, :, :, :, None, :].expand_as(props)
            # - add a dim for the MVNs

            weights = props * weights
            weights = weights.sum(dim=5)  # - sum out the MVNs

            assert indices.size() == (b, h, w, k, vs, 2)
            assert weights.size() == (b, h, w, k, vs)

        else:
            vs = 1
            indices = means.floor().to(torch.long).detach()

        l = h * w * k * vs
        indices = indices.view(b*l, 2)

        br = torch.arange(b, device=d(x), dtype=torch.long)[:, None].expand(b, l).contiguous().view(-1)
        features = x[br, :, indices[:, 0], indices[:, 1]]
        assert features.size() == (b*l, c)

        if self.smp:
            features = features.view(b, h, w, k, vs, c)
            features = features * weights[:, :, :, :, :, None]
            features = features.sum(dim=4)
        else:
            features = features.view(b, h, w, k, c)

        # features now contains the selected input pixels (or weighted sum thereover): k inputs per output pixel
        assert features.size() == (b, h, w, k, c), f'Was {features.size()}, expected {(b, h, w, k, c)}.'

        features = features.view(b, h, w, k * c)

        return self.unify(features).permute(0, 3, 1, 2) # (b, c_out, h, w)

    def plot(self, images, numpixels=5):

        b, c, h, w = images.size()
        k = self.k

        # choose 5 random pixels, for which we'll plot the input pixels.
        choices = torch.randint(low=0, high=h*w, size=(numpixels,))

        perrow = 5

        rows = int(math.ceil(b/perrow))

        means, sigmas, _ = self.hyper(images)

        images = images.data

        plt.figure(figsize=(perrow * 3, rows*3))

        for current in range(b):

            # select subset of means, sigmas
            smeans = means[current, :, :, :, :].view(h*w, k, 2)
            ssigmas = sigmas[current, :, :, :].view(h*w, k, 2)
            color = (torch.arange(numpixels, dtype=torch.float)[:, None].expand(numpixels, k)/numpixels) * 2.0 - 1.0

            smeans = smeans[choices, :, :]
            ssigmas = ssigmas[choices, :]

            ax = plt.subplot(rows, perrow, current+1)

            im = np.transpose(images[current, :, :, :].cpu().numpy(), (1, 2, 0))
            im = np.squeeze(im)

            ax.imshow(im, interpolation='nearest', extent=(-0.5, w-0.5, -0.5, h-0.5), cmap='gray_r')

            util.plot(smeans.reshape(1, -1, 2), ssigmas.reshape(1, -1, 2), color.reshape(1, -1), axes=ax, flip_y=h, tanh=False)

        plt.gcf()

class DavidLayer(nn.Module):

    def __init__(self, cin, cout, **kwargs):
        super().__init__()

        self.block0 = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU()
        )

    def forward(self, x):

        x = self.block0(x)

        return x + self.block2(self.block1(x))

class DavidNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.layer0 = DavidLayer(64, 128)

        self.mid = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1 = DavidLayer(256, 512)

        self.head = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=1),
            util.Flatten(),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):

        x = self.prep(x)
        x = self.layer0(x)
        x = self.mid(x)
        x = self.layer1(x)
        x = self.head(x)

        return x

class Classifier(nn.Module):

    def __init__(self, in_size, num_classes, **kwargs):
        super().__init__()

        c, h, w = in_size

        #self.layer0 = nn.Conv2d(c, c, kernel_size=3, padding=1)

        self.sparse = Convolution((c, h, w), (32, h, w), **kwargs)
        # self.nsp = nn.Conv2d(c, 32, kernel_size=3, padding=1)

        self.blocks = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), # 16x16
            MobBlock(32),               nn.Conv2d(32, 16, kernel_size=1),
            MobBlock(16), MobBlock(16), nn.Conv2d(16, 24, kernel_size=1),
            nn.MaxPool2d(kernel_size=2), # 8x8
            MobBlock(24, kernel=5), MobBlock(24, kernel=5), nn.Conv2d(24, 40, kernel_size=1),
            nn.MaxPool2d(kernel_size=2), # 4x4
            MobBlock(40), MobBlock(40), MobBlock(40), nn.Conv2d(40, 80, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2),
            MobBlock(80, kernel=5), MobBlock(80, kernel=5), MobBlock(80, kernel=5), nn.Conv2d(80, 112, kernel_size=1),
            #nn.MaxPool2d(kernel_size=2),
            MobBlock(112, kernel=5), MobBlock(112, kernel=5), MobBlock(112, kernel=5), MobBlock(112, kernel=5), nn.Conv2d(112, 192, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2),
            MobBlock(192), nn.Conv2d(192, 320, kernel_size=1),
            util.Flatten(),
            nn.Linear(320 * 4 * 4, num_classes)
        )


    def forward(self, x):

        x = self.sparse(x)
        x = self.blocks(x)

        return x

class MiniClassifier(nn.Module):

    def __init__(self, in_size, num_classes, **kwargs):
        super().__init__()

        c, h, w = in_size

        self.blocks = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=1), MobBlock(32),
            nn.MaxPool2d(kernel_size=2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=1), MobBlock(64),
            nn.MaxPool2d(kernel_size=2),  # 8x8
            nn.Conv2d(64, 128, kernel_size=1), MobBlock(128),
            nn.MaxPool2d(kernel_size=2), # 4x4
            MobBlock(128), nn.Conv2d(128, 320, kernel_size=1),
            nn.MaxPool2d(kernel_size=2),  # 2x2
            # ResBlock(64), nn.Conv2d(64, 112, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2),  # 4x4
            # ResBlock(112, kernel=5), nn.Conv2d(112, 192, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2), # 2x2
            # ResBlock(192), nn.Conv2d(192, 320, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2), # 1x1
            util.Flatten(),
            nn.Linear(320 * 2 * 2, num_classes)
        )

    def forward(self, x):

        return self.blocks(x)

class ThreeCClassifier(nn.Module):

    def __init__(self, in_size, num_classes, **kwargs):
        super().__init__()

        c, h, w = in_size

        self.sparse = Convolution((c, h, w), (32, h, w), **kwargs)
        self.spars1 = Convolution((32, h//2, w//2), (64, h, w), **kwargs)
        self.spars2 = Convolution((64, h//4, w//4), (128, h, w), **kwargs)

        self.blocks = nn.Sequential(
            self.sparse, nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16x16
            self.spars1, nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 8x8
            self.spars2, nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 4x4
            MobBlock(128), nn.Conv2d(128, 320, kernel_size=1),
            nn.MaxPool2d(kernel_size=2), # 2x2
            util.Flatten(),
            nn.Linear(320 * 2 * 2, num_classes)
        )


    def forward(self, x):

        return self.blocks(x)

def go(arg):

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    normalize = transforms.Compose([transforms.ToTensor()])

    if (arg.task == 'mnist'):
        data = arg.data + os.sep + arg.task

        if arg.final:
            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=normalize)
            trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch_size, shuffle=True, num_workers=2)

            test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=normalize)
            testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch_size, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=normalize)

            trainloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

        shape = (1, 28, 28)
        num_classes = 10

    elif (arg.task == 'cifar10'):

        data = arg.data + os.sep + arg.task

        if arg.final:
            train = torchvision.datasets.CIFAR10(root=data, train=True, download=True, transform=ToTensor())
            trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch_size, shuffle=True, num_workers=2)
            test = torchvision.datasets.CIFAR10(root=data, train=False, download=True, transform=ToTensor())
            testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch_size, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.CIFAR10(root=data, train=True, download=True, transform=ToTensor())

            trainloader = DataLoader(train, batch_size=arg.batch_size, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=arg.batch_size,
                                    sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

            shape = (3, 32, 32)
            num_classes = 10
    else:
        raise Exception('Task {} not recognized'.format(arg.task))

    # Create model
    if arg.model == 'efficient':
        model = Classifier(shape, num_classes, k=arg.k, gadditional=arg.gadditional, radditional=arg.radditional, region=arg.region,
                       adaptive=arg.adaptive, sigma_scale=arg.sigma_scale, modulo=arg.modulo)
        sparse = True
    elif arg.model == 'mini':
        model = MiniClassifier(shape, num_classes, k=arg.k, gadditional=arg.gadditional, radditional=arg.radditional, region=arg.region,
                       adaptive=arg.adaptive, sigma_scale=arg.sigma_scale, modulo=arg.modulo)
        sparse = False
    elif arg.model == '3c':
        model = ThreeCClassifier(shape, num_classes, k=arg.k, gadditional=arg.gadditional, radditional=arg.radditional, region=arg.region,
                       adaptive=arg.adaptive, sigma_scale=arg.sigma_scale, modulo=arg.modulo)
        sparse = True

    elif arg.model == 'david':
        model = DavidNet(num_classes)
        sparse = False
    else:
        raise Exception(f'Model {arg.model} not recognized')


    if arg.cuda:
        model.cuda()

    opt = torch.optim.Adam(params=model.parameters(), lr=arg.lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i : min(  i/arg.lr_warmup, 1.0) )

    # Training loop
    util.makedirs(f'./{arg.task}/')
    seen = 0
    for e in range(arg.epochs):

        model.train(True)

        for i, (inputs, labels) in enumerate(tqdm.tqdm(trainloader, 0)):

            b, c, h, w = inputs.size()
            seen += b

            if sparse:
                model.sparse.sample(random.random() < arg.sample_prob) # sample every tenth batch

            if arg.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            opt.zero_grad()

            outputs = model(inputs)

            loss = F.cross_entropy(outputs, labels)

            loss.backward()

            # clip gradients
            if arg.gradient_clipping is not None:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()

            tbw.add_scalar('sparsity/loss', loss.item()/b, seen)

            sch.step()

            if sparse and i == 0 and e % arg.plot_every == 0:
                model.sparse.plot(inputs[:10, ...])
                plt.savefig(f'{arg.task}/convolution.{e:03}.pdf')

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

                total += outcls.size(0)
                correct += (outcls == labels).sum().item()

            acc = correct / float(total)

            print('\nepoch {}: {}\n'.format(e, acc))
            tbw.add_scalar('sparsity/test acc', acc, e)


if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs.",
                        default=80, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of input  per output pixel.",
                        default=9, type=int)

    parser.add_argument("-a", "--gadditional",
                        dest="gadditional",
                        help="Number of additional points sampled globally",
                        default=8, type=int)

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Which model (efficient, mini, 3c, david)",
                        default='efficient', type=str)

    parser.add_argument("-A", "--radditional",
                        dest="radditional",
                        help="Number of additional points sampled locally",
                        default=8, type=int)

    parser.add_argument("-R", "--region",
                        dest="region",
                        help="Size of the (square) region to use for local sampling.",
                        default=0.25, type=float)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-t", "--task", dest="task",
                        help="Dataset (cifar10)",
                        default='cifar10')

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Minimum value of sigma.",
                        default=0.01, type=float)

    parser.add_argument("-S", "--sigma-scale",
                        dest="sigma_scale",
                        help="Scalar applied to sigmas.",
                        default=0.5, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)

    parser.add_argument("-D", "--data-dir", dest="data",
                        help="Data directory",
                        default='./data')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many epochs between tests.",
                        default=5, type=int)

    parser.add_argument("--plot-every",
                        dest="plot_every",
                        help="How many epochs between plotting the sparse indices.",
                        default=1, type=int)

    parser.add_argument("--adaptive", dest="adaptive",
                        help="Whether to base the index tuple structure on the pixel representation in the previous layer.",
                        action="store_true")

    parser.add_argument("--modulo", dest="modulo",
                        help="Use modulo operator to fit continuous index tuples to the required range.",
                        action="store_true")
    #
    # parser.add_argument("--partial-loss", dest="ploss",
    #                     help="Use only the last element of the sequence for the loss.",
    #                     action="store_true")

    parser.add_argument("--sample-prob",
                        dest="sample_prob",
                        help="Sample probability (with this probability we sample index tuples).",
                        default=0.5, type=float)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=100000, type=int)


    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
