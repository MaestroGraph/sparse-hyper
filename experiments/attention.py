from _context import sparse

from sparse import util
from util import Lambda, Debug
from util import od, prod, logit

import time,  os, math, sys, PIL
import torch, random
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter

from skimage.transform import resize

from math import *
from torch.utils.data import TensorDataset, DataLoader

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100


from argparse import ArgumentParser

import logging

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG = logging.getLogger('ash')
LOG.setLevel(logging.INFO)
fh = logging.FileHandler('ash.log')
fh.setLevel(logging.INFO)
LOG.addHandler(fh)

SIGMA_BOOST_REINFORCE = 2.0 # ensure that first sigmas are large enough
EPSILON = 10e-7

def inv(i, max):
    sc = (i/max) * 0.999 + 0.0005
    return logit(sc)

def sigmoid(x):
    if type(x) == float:
        return 1 / (1 + math.exp(-x))
    return 1 / (1 + torch.exp(-x))

def rescale(image, outsize):
    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    image = resize(image, outsize)

    return torch.from_numpy( np.transpose(image, (2, 0, 1)) )

HIDLIN = 512
def prep(ci, hi, wi):
    """
    Canonical preprocessing model (list of modules). Results in linear layer
    of HIDLIN units

    :return:
    """
    activation = nn.ReLU()

    p1, p2 = 4, 2
    ch1, ch2, ch3 = 32, 64, 128

    hid = max(1, floor(floor(wi / p1) / p2) * floor(floor(hi / p1) / p2)) * ch3

    return [
        nn.Conv2d(ci, ch1, kernel_size=3, padding=1),
        activation,
        nn.Conv2d(ch1, ch1, kernel_size=3, padding=1),
        activation,
        nn.MaxPool2d(kernel_size=p1),
        nn.Conv2d(ch1, ch2, kernel_size=3, padding=1),
        activation,
        nn.Conv2d(ch2, ch2, kernel_size=3, padding=1),
        activation,
        nn.MaxPool2d(kernel_size=p2),
        nn.Conv2d(ch2, ch3, kernel_size=3, padding=1),
        activation,
        nn.Conv2d(ch3, ch3, kernel_size=3, padding=1),
        activation,
        util.Flatten(),
        nn.Linear(hid, HIDLIN),
        activation,
        nn.Linear(HIDLIN, HIDLIN)
    ]

class STNAttentionLayer(nn.Module):
    """
    Baseline: spatial transformer
    """

    def __init__(self, in_size, k, glimpses=1, scale=0.001):
        super().__init__()

        self.in_size = in_size
        self.k = k
        self.num_glimpses = glimpses
        self.scale=scale

        ci, hi, wi = in_size
        co, ho, wo = ci , k, k

        modules = prep(ci, hi, wi) + [nn.ReLU(), nn.Linear(HIDLIN, 3 * 2 * glimpses), util.Reshape((glimpses, 2, 3))]
        self.preprocess = nn.Sequential(*modules)

        self.register_buffer('identity', torch.FloatTensor([1, 0, 0, 0, 1, 0]).view(2, 3))

    def forward(self, image):

        b, ci, hi, wi = image.size()
        thetas = self.preprocess(image)

        # ensure that the starting point is close enough to the identity transform
        thetas = thetas * self.scale + self.identity[None, None, :, :]

        b, g, _, _ = thetas.size()

        grid = F.affine_grid(thetas.view(b*g, 2, 3), torch.Size((b*g, self.in_size[0], self.k, self.k)) )

        out = F.grid_sample(
            image[:, None, :, :, :].expand(b, g, ci, hi, wi).contiguous().view(b*g, ci, hi, wi),
            grid)
        bg, co, ho, wo = out.size()

        return out.view(b, g, co, ho, wo)

    def plot(self, images):
        perrow = 5

        num, c, w, h = images.size()

        rows = int(math.ceil(num/perrow))

        thetas = self.preprocess(images)
        thetas = thetas * self.scale + self.identity[None, None, :, :]

        b, g, _, _ = thetas.size()
        grid = F.affine_grid( thetas.view(b*g, 2, 3), torch.Size((b*g, self.in_size[0], self.k, self.k)) )
        means = grid.view(b, -1, 2).data.cpu()

        # scale to image resolution
        means = ((means + 1.0) * 0.5) * torch.tensor(self.in_size[1:], dtype=torch.float)[None, None, :]

        b, k, _ = means.size()
        sigmas = torch.ones((b, k, 2)) * 0.001
        values = torch.ones((b, k))

        images = images.data

        plt.figure(figsize=(perrow * 3, rows*3))

        for i in range(num):

            ax = plt.subplot(rows, perrow, i+1)

            im = np.transpose(images[i, :, :, :].cpu().numpy(), (1, 2, 0))
            im = np.squeeze(im)

            ax.imshow(im, interpolation='nearest', extent=(-0.5, w-0.5, -0.5, h-0.5), cmap='gray_r')

            util.plot(means[i, :, :].unsqueeze(0), sigmas[i, :, :].unsqueeze(0), values[i, :].unsqueeze(0), axes=ax, flip_y=h, alpha_global=0.8/self.num_glimpses)

        plt.gcf()

class BoxAttentionLayer(sparse.SparseLayer):
    """
    NB: k is the number of tuples per input dimension. That is, k = 4 results in a 16 * c grid of inputs evenly spaced
     across a bounding box
    """

    def __init__(self, in_size, k,  gadditional=0, radditional=0,
                 region=None, sigma_scale=0.1,
                 num_values=-1, min_sigma=0.0, glimpses=1):

        assert(len(in_size) == 3)

        self.in_size = in_size
        self.k = k
        self.sigma_scale = sigma_scale
        self.num_values = num_values
        self.min_sigma = min_sigma
        self.num_glimpses = glimpses

        ci, hi, wi = in_size
        co, ho, wo = ci , k, k
        out_size = glimpses, co, ho, wo

        self.out_size = out_size

        map = (glimpses, co, k, k)

        template = torch.LongTensor(list(np.ndindex( map))) # [:, (2, 0, 1)]

        assert template.size() == (prod(map), 4)

        pixel_indices = template[:, 2:].clone()

        template = torch.cat([template, template[:, 1:]], dim=1)

        assert template.size() == (prod(map), 7)

        self.lc = [5, 6] # learnable columns

        super().__init__(
            in_rank=3, out_size=(glimpses, co, ho, wo),
            temp_indices=template,
            learn_cols=self.lc,
            chunk_size=1,
            gadditional=gadditional, radditional=radditional, region=region,
            bias_type=util.Bias.NONE)

        # scale to [0,1] in each dim
        pixel_indices = pixel_indices.float() / torch.FloatTensor([[k, k]]).expand_as(pixel_indices)
        self.register_buffer('pixel_indices', pixel_indices)

        modules = prep(ci, hi, wi) + [nn.ReLU(), nn.Linear(HIDLIN, 4 * glimpses), util.Reshape((glimpses, 4))]
        self.preprocess = nn.Sequential(*modules)

        self.register_buffer('bbox_offset', torch.FloatTensor([-1, 1, -1, 1]))
        # -- added to the bounding box, to make sure there's a training signal
        #    from the initial weights (i.e. in case all outputs are close to zero)

        # One sigma per glimpse
        self.sigmas = Parameter(torch.randn( (glimpses, ) ))

        # All values 1, no bias. Glimpses extract only pixel information.
        self.register_buffer('one', torch.FloatTensor([1.0]))

    def hyper(self, input, prep=None):
        """
        Evaluates hypernetwork.
        """

        b, c, h, w = input.size()

        bboxes = self.preprocess(input) # (b, g, 4)
        b, g, _ = bboxes.size()

        # ensure that the bounding box covers a reasonable area of the image at the start
        bboxes = bboxes + self.bbox_offset[None, None, :]

        # Fit to the max pixel values
        bboxes = sparse.transform_means(bboxes, (h, h, w, w))

        vmin, vmax, hmin, hmax = bboxes[:, :, 0], bboxes[:, :, 1], bboxes[:, :, 2], bboxes[:, :, 3] # vert (height), hor (width),

        vrange, hrange = vmax - vmin, hmax - hmin

        pih, _ = self.pixel_indices.size()
        pixel_indices = self.pixel_indices.view(g, pih // g, 2)
        pixel_indices = pixel_indices[None, :, :, :].expand(b, g, pih // g, 2)

        range = torch.cat([vrange[:, :, None, None], hrange[:, :, None, None]], dim=3)
        range = range.expand(b, g, pih//g, 2)

        min = torch.cat([vmin[:, :, None, None], hmin[:, :, None,  None]], dim=3)
        min = min.expand(b, g, pih//g, 2)

        means = pixel_indices * range + min
        means = means.view(b, pih, 2)

        # Expand sigmas
        sigmas = self.sigmas[None, :, None].expand(b, g, pih//g).contiguous().view(b, pih)
        sigmas = sparse.transform_sigmas(sigmas, (h, w))
        sigmas = sigmas * self.sigma_scale + self.min_sigma

        values = self.one[None, :].expand(b, pih)

        return means, sigmas, values

    def plot(self, images):
        perrow = 5

        num, c, w, h = images.size()

        rows = int(math.ceil(num/perrow))

        means, sigmas, values = self.hyper(images)

        images = images.data

        plt.figure(figsize=(perrow * 3, rows*3))

        for i in range(num):

            ax = plt.subplot(rows, perrow, i+1)

            im = np.transpose(images[i, :, :, :].cpu().numpy(), (1, 2, 0))
            im = np.squeeze(im)

            ax.imshow(im, interpolation='nearest', extent=(-0.5, w-0.5, -0.5, h-0.5), cmap='gray_r')

            util.plot(means[i, :, :].unsqueeze(0), sigmas[i, :, :].unsqueeze(0), values[i, :].unsqueeze(0), axes=ax, flip_y=h, alpha_global=0.8/self.num_glimpses)

        plt.gcf()

class QuadAttentionLayer(sparse.SparseLayer):
    """
    Version of the attention layer that uses an attention _quadrangle_, instead of a bounding box.

    NB: k is the number of tuples per input dimension. That is, k = 4 results in a 16 * c grid of inputs evenly spaced
     across a bounding box


    NOTE: May not work for color images... not properly tested yet.
    """

    def __init__(self, in_size, k,  gadditional=0, radditional=0,
                 region=None, sigma_scale=0.1,
                 num_values=-1, min_sigma=0.0, glimpses=1):

        assert(len(in_size) == 3)

        self.in_size = in_size
        self.k = k
        self.sigma_scale = sigma_scale
        self.num_values = num_values
        self.min_sigma = min_sigma

        ci, hi, wi = in_size
        co, ho, wo = ci , k, k
        out_size = glimpses, co, ho, wo

        self.out_size = out_size

        map = (glimpses, co, k, k)

        template = torch.LongTensor(list(np.ndindex( map)))

        assert template.size() == (prod(map), 4)

        template = torch.cat([template, template[:, 1:]], dim=1)

        assert template.size() == (prod(map), 7)

        self.lc = [5, 6] # learnable columns

        super().__init__(
            in_rank=3, out_size=(glimpses, co, ho, wo),
            temp_indices=template,
            learn_cols=self.lc,
            chunk_size=1,
            gadditional=gadditional, radditional=radditional, region=region,
            bias_type=util.Bias.NONE)

        self.num_glimpses = glimpses

        modules = prep(ci, hi, wi) + [nn.ReLU(), nn.Linear(HIDLIN, 8 * glimpses), util.Reshape((glimpses, 4, 2))]
        self.preprocess = nn.Sequential(*modules)

        self.register_buffer('grid', util.interpolation_grid((k, k)))

        self.register_buffer('quad_offset', torch.FloatTensor([[-1, 1], [1, 1], [1, -1], [-1, -1]]))
        # -- added to the quad, to make sure there's a training signal
        #    from the initial weights (i.e. in case all outputs are close to zero)

        # One sigma per glimpse
        self.sigmas = Parameter(torch.randn( (glimpses, ) ))

        # All values 1, no bias. Glimpses extract only pixel information.
        self.register_buffer('one', torch.FloatTensor([1.0]))

    def hyper(self, input, prep=None):
        """
        Evaluates hypernetwork.
        """

        b, c, h, w = input.size()

        quad = self.preprocess(input) # Cpompute the attention quadrangle
        b, g, _, _ = quad.size() # (b, g, 4, 2)
        k, k, _ = self.grid.size() # k, k, 4

        # ensure that the bounding box covers a reasonable area of the image at the start
        quad = quad + self.quad_offset[None, None, :]

        # Fit to the max pixel values
        quad = sparse.transform_means(quad.view(b, g*4, 2), (h, w)).view(b, g, 4, 2)

        # Interpolate between the four corners of the quad
        grid = self.grid[None, None, :,    :,    :, None] # b, g, k, k, 4, 2
        quad =      quad[:,    :,    None, None, :, :]

        res = (grid * quad).sum(dim=4)

        assert res.size() == (b, g, k, k, 2)

        means = res.view(b, g * k * k, 2)

        # Expand sigmas
        sigmas = self.sigmas[None, :, None].expand(b, g, (k*k)).contiguous().view(b, (g*k*k))
        sigmas = sparse.transform_sigmas(sigmas, (h, w))
        sigmas = sigmas * self.sigma_scale + self.min_sigma

        values = self.one[None, :].expand(b, k*k*g)

        return means, sigmas, values

    def plot(self, images):
        perrow = 5

        num, c, w, h = images.size()

        rows = int(math.ceil(num/perrow))

        means, sigmas, values = self.hyper(images)

        images = images.data

        plt.figure(figsize=(perrow * 3, rows*3))

        for i in range(num):

            ax = plt.subplot(rows, perrow, i+1)

            im = np.transpose(images[i, :, :, :].cpu().numpy(), (1, 2, 0))
            im = np.squeeze(im)

            ax.imshow(im, interpolation='nearest', extent=(-0.5, w-0.5, -0.5, h-0.5), cmap='gray_r')

            util.plot(means[i, :, :].unsqueeze(0), sigmas[i, :, :].unsqueeze(0), values[i, :].unsqueeze(0), axes=ax, flip_y=h, alpha_global=0.8/self.num_glimpses)

        plt.gcf()

class ReinforceLayer(nn.Module):
    """
    Simple reinforce-based baseline. Assumes a fixed glimpse size.
    """

    def __init__(self, in_shape, glimpses, glimpse_size,
                 num_classes, rfboost=2.0):
        super().__init__()

        self.rfboost = rfboost
        self.num_glimpses = glimpses
        self.glimpse_size = glimpse_size
        self.in_shape = in_shape

        activation = nn.ReLU()

        ci, hi, wi = in_shape

        modules = prep(ci, hi, wi) + [nn.ReLU(), nn.Linear(HIDLIN, glimpses * 3), util.Reshape((glimpses, 3))]
        self.preprocess = nn.Sequential(*modules)

    def forward(self, image):

        prep = self.preprocess(image)
        b, g, _= prep.size()
        ci, hi, wi = self.in_shape
        hg, wg = self.glimpse_size

        means = prep[:, :, :2]
        sigmas = prep[:, : , 2]

        sigmas = sparse.transform_sigmas(sigmas, self.in_shape[1:])

        stoch_means = torch.distributions.Normal(means, sigmas)
        sample = stoch_means.sample()

        point_means = sparse.transform_means(sample, (hi-hg, wi-wg)).round().long()

        # extract
        batch = []
        for bi in range(b):
            extracts = []
            for gi in range(g):
                h, w = point_means[bi, gi, :]
                ext = image.data[bi, :, h:h+hg, w:w+wg]
                extracts.append(ext[None, None, :, :, :])

            batch.append(torch.cat(extracts, dim=1))
        result = torch.cat(batch, dim=0)

        return result, stoch_means, sample

    def plot(self, images):
        perrow = 5

        num, c, w, h = images.size()

        rows = int(math.ceil(num/perrow))

        prep = self.preprocess(images)
        b, g, _= prep.size()
        ci, hi, wi = self.in_shape
        hg, wg = self.glimpse_size

        means = prep[:, :, :2]
        means = sparse.transform_means(means, (hi-hg, wi-wg)).round().long()
        sigmas = prep[:, : , 2]
        sigmas = sparse.transform_sigmas(sigmas, self.in_shape[1:])

        images = images.data

        plt.figure(figsize=(perrow * 3, rows*3))

        for i in range(num):

            ax = plt.subplot(rows, perrow, i+1)

            im = np.transpose(images[i, :, :, :].cpu().numpy(), (1, 2, 0))
            im = np.squeeze(im)

            ax.imshow(im, interpolation='nearest', extent=(-0.5, w-0.5, -0.5, h-0.5), cmap='gray_r')

            util.plot(means[i, :, :].unsqueeze(0), sigmas[i, :, :].unsqueeze(0), torch.ones(means[:, :, 0].size()), axes=ax, flip_y=h, alpha_global=0.8/self.num_glimpses)

        plt.gcf()

class R(nn.Module):
    """
    Helper module for reinforcement learning pipelines. Passes extra arguments along down the pipeline
    """
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, tuple):
        x, rest = tuple[0], tuple[1:]

        out = self.inner(x)
        return (out,) + rest

PLOT = True
COLUMN = 13

def go(arg):

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    normalize = transforms.Compose([transforms.ToTensor()])

    if(arg.task=='mnist'):
        data = arg.data + os.sep + arg.task

        if arg.final:
            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=normalize)
            trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch, shuffle=True, num_workers=2)

            test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=normalize)
            testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=normalize)

            trainloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

        shape = (1, 28, 28)
        num_classes = 10

    elif (arg.task == 'image-folder-bw'):

        tr = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

        if arg.final:
            train = torchvision.datasets.ImageFolder(root=arg.data + '/train/', transform=tr)
            test  = torchvision.datasets.ImageFolder(root=arg.data + '/test/', transform=tr)

            trainloader = DataLoader(train, batch_size=arg.batch, shuffle=True)
            testloader = DataLoader(train, batch_size=arg.batch, shuffle=True)

        else:

            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.ImageFolder(root=arg.data + '/train/', transform=tr)

            trainloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=arg.batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))


        for im, labels in trainloader:
            shape = im[0].size()
            break

        num_classes = 10

    else:
        raise Exception('Task name {} not recognized'.format(arg.task))

    activation = nn.ReLU()

    hyperlayer = None

    if arg.modelname == 'conv':

        base = prep(*shape)

        model = nn.Sequential(*(
            base +
            [activation, nn.Linear(HIDLIN, num_classes),
            nn.Softmax()])
        )

        reinforce = False

    elif arg.modelname == 'reinforce':

        hyperlayer = ReinforceLayer(in_shape=shape, glimpses=arg.num_glimpses,
                glimpse_size=(28, 28),
                num_classes=num_classes)

        model = nn.Sequential(
             hyperlayer,
             R(util.Flatten()),
             R(nn.Linear(28 * 28 * shape[0] * arg.num_glimpses, arg.hidden)),
             R(activation),
             R(nn.Linear(arg.hidden, num_classes)),
             R(nn.Softmax())
        )

        reinforce = True

    elif arg.modelname == 'ash':

        hyperlayer = BoxAttentionLayer(
            glimpses=arg.num_glimpses,
            in_size=shape, k=arg.k,
            gadditional=arg.gadditional, radditional=arg.radditional, region=(arg.region, arg.region),
            min_sigma=arg.min_sigma
        )

        model = nn.Sequential(
             hyperlayer,
             util.Flatten(),
             nn.Linear(arg.k * arg.k * shape[0] * arg.num_glimpses, arg.hidden),
             activation,
             nn.Linear(arg.hidden, num_classes),
             nn.Softmax()
        )

        reinforce = False

    elif arg.modelname == 'quad':
        """
        Network with quadrangle attention (instead of bounding box).
        """

        hyperlayer = QuadAttentionLayer(
            glimpses=arg.num_glimpses,
            in_size=shape, k=arg.k,
            gadditional=arg.gadditional, radditional=arg.radditional, region=(arg.region, arg.region),
            min_sigma=arg.min_sigma
        )

        model = nn.Sequential(
             hyperlayer,
             util.Flatten(),
             nn.Linear(arg.k * arg.k * shape[0] * arg.num_glimpses, arg.hidden),
             activation,
             nn.Linear(arg.hidden, num_classes),
             nn.Softmax()
        )

        reinforce = False

    elif arg.modelname == 'stn':
        """
        Spatial transformer with an MLP head.
        """

        hyperlayer = STNAttentionLayer(in_size=shape, k=arg.k, glimpses=arg.num_glimpses, scale=arg.stn_scale)

        model = nn.Sequential(
             hyperlayer,
             util.Flatten(),
             nn.Linear(arg.k * arg.k * shape[0] * arg.num_glimpses, arg.hidden),
             activation,
             nn.Linear(arg.hidden, num_classes),
             nn.Softmax()
        )

        reinforce = False

    elif arg.modelname == 'stn-conv':
        """
        Spatial transformer with a convolutional head.
        """

        hyperlayer = STNAttentionLayer(in_size=shape, k=arg.k, glimpses=arg.num_glimpses, scale=arg.stn_scale)

        ch1, ch2, ch3 = 16, 32, 64
        h = (arg.k // 8) ** 2 * 64

        model = nn.Sequential(
            hyperlayer,
            util.Reshape((arg.num_glimpses * shape[0], arg.k, arg.k)), # Fold glimpses into channels
            nn.Conv2d(arg.num_glimpses * shape[0], ch1, kernel_size=3, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1),
            activation,
            nn.Conv2d(ch2, ch2, kernel_size=3, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(ch2, ch3, kernel_size=3, padding=1),
            activation,
            nn.Conv2d(ch3, ch3, kernel_size=3, padding=1),
            activation,
            nn.MaxPool2d(kernel_size=2),
            util.Flatten(),
            nn.Linear(h, 128),
            activation,
            nn.Linear(128, num_classes),
            nn.Softmax()
        )

        reinforce = False

    elif arg.modelname == 'ash-conv':
        """
        Model with a convolution head. More powerful classification, but more difficult to train on top of a hyperlayer.
        """

        hyperlayer = BoxAttentionLayer(
            glimpses=arg.num_glimpses,
            in_size=shape, k=arg.k,
            gadditional=arg.gadditional, radditional=arg.radditional, region=(arg.region, arg.region),
            min_sigma=arg.min_sigma
        )

        ch1, ch2, ch3 = 16, 32, 64
        h = (arg.k // 8) ** 2 * 64

        model = nn.Sequential(
            hyperlayer,
            util.Reshape((arg.num_glimpses * shape[0], arg.k, arg.k)), # Fold glimpses into channels
            nn.Conv2d(arg.num_glimpses * shape[0], ch1, kernel_size=5, padding=2),
            activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(ch1, ch2, kernel_size=5, padding=2),
            activation,
            nn.Conv2d(ch2, ch2, kernel_size=5, padding=2),
            activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(ch2, ch3, kernel_size=5, padding=2),
            activation,
            nn.Conv2d(ch3, ch3, kernel_size=5, padding=2),
            activation,
            nn.MaxPool2d(kernel_size=2),
            util.Flatten(),
            nn.Linear(h, 128),
            activation,
            nn.Linear(128, num_classes),
            nn.Softmax()
        )

        reinforce = False

    else:
        raise Exception('Model name {} not recognized'.format(arg.modelname))

    if arg.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=arg.lr)

    xent = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    step = 0

    sigs, vals = [], []

    util.makedirs('./mnist/')

    for epoch in range(arg.epochs):

        model.train(True)

        for i, (inputs, labels) in tqdm(enumerate(trainloader, 0)):

            # if i> 2:
            #     break

            if arg.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            if not reinforce:
                outputs = model(inputs)
            else:
                outputs, stoch_nodes, actions = model(inputs)

            mloss = F.cross_entropy(outputs, labels, reduce=False)

            if reinforce:

                rloss = stoch_nodes.log_prob(actions) * - mloss.detach()[:, None, None]

                loss = rloss.sum(dim=1) + mloss[:, None]

                tbw.add_scalar('mnist/train-loss', float(loss.mean().item()), step)
                tbw.add_scalar('mnist/model-loss', float(rloss.sum(dim=1).mean().item()), step)
                tbw.add_scalar('mnist/reinf-loss', float(mloss.mean().item()), step)

            else:
                loss = mloss

                tbw.add_scalar('mnist/train-loss', float(loss.data.sum().item()), step)

            loss = loss.sum()
            loss.backward()  # compute the gradients

            optimizer.step()

            step += inputs.size(0)

            if epoch % arg.plot_every == 0 and i == 0 and hyperlayer is not None:

                hyperlayer.plot(inputs[:10, ...])
                plt.savefig('mnist/attention.{:03}.pdf'.format(epoch))

        total = 0.0
        correct = 0.0

        model.train(False)

        for i, (inputs, labels) in enumerate(testloader, 0):

            if arg.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            if not reinforce:
                outputs = model(inputs)
            else:
                outputs, _, _ = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct/total

        tbw.add_scalar('mnist1d/per-epoch-test-acc', accuracy, epoch)
        print('EPOCH {}: {} accuracy '.format(epoch, accuracy))

    LOG.info('Finished Training.')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs over the generated data.",
                        default=350, type=int)


    parser.add_argument("-p", "--plot-every",
                        dest="plot_every",
                        help="Plot every n epochs.",
                        default=1, type=int)

    parser.add_argument("-m", "--model",
                        dest="modelname",
                        help="Which model to train.",
                        default='baseline')

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples in the decoder layer",
                        default=3, type=int)

    parser.add_argument("-a", "--gadditional",
                        dest="gadditional",
                        help="Number of additional points sampled globally",
                        default=8, type=int)

    parser.add_argument("-A", "--radditional",
                        dest="radditional",
                        help="Number of additional points sampled locally",
                        default=8, type=int)

    parser.add_argument("-R", "--region",
                        dest="region",
                        help="Size of the (square) region to use for local sampling.",
                        default=8, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data/')

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-S", "--subsample",
                        dest="subsample",
                        help="Sample a subset of the indices to estimate gradients for",
                        default=None, type=float)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Minimum value of sigma.",
                        default=0.0, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)

    parser.add_argument("-t", "--task", dest="task",
                        help="Task (mnist, cifar10, cifar100)",
                        default='mnist')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set.",
                        action="store_true")

    parser.add_argument("-H", "--hidden", dest="hidden",
                        help="Size of the hidden layer.",
                        default=32, type=int)

    parser.add_argument("-Q", "--dropout", dest="dropout",
                        help="Dropout of the baseline and hypernetwork.",
                        default=0.0, type=float)

    parser.add_argument("--reinforce-boost", dest="rfboost",
                        help="boost the means of the reinforce method.",
                        default=2.0, type=float)

    parser.add_argument("-G", "--num-glimpses", dest="num_glimpses",
                        help="Number of glimpses for the ash model.",
                        default=4, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--stn-scale", dest="stn_scale",
                        help="Scaling parameter to fintetune the initialiazation of the STN network.",
                        default=0.01, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(options)
