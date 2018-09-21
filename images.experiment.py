import gaussian, gaussian_temp, util, time,  os, math, sys, PIL
import torch, random
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from util import Lambda, Debug
from torch.utils.serialization import load_lua

from skimage.transform import resize

from math import *
from torch.utils.data import TensorDataset, DataLoader

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

from util import od, prod, logit

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

DROPOUT = 0.0

def inv(i, max):
    sc = (i/max) * 0.999 + 0.0005
    return logit(sc)

def sigmoid(x):
    if type(x) == float:
        return 1 / (1 + math.exp(-x))
    return 1 / (1 + torch.exp(-x))
#
# class ImageLayer(gaussian_temp.HyperLayer):
#     """
#     Simple hyperlayer for the 1D MNIST experiment
#
#     NB: k is the number of tuples _per hidden node_.
#     """
#
#     def __init__(self, in_size, out_size, k, adaptive=True, additional=0, sigma_scale=0.1, num_values=-1, min_sigma=0.0, pre=0, subsample=None, mix=False):
#
#         out_indices = torch.LongTensor(list(np.ndindex(out_size)))
#
#         out_indices = out_indices.unsqueeze(1).expand(prod(out_size), k, len(out_size))
#         out_indices = out_indices.contiguous().view(prod(out_size) * k, len(out_size))
#
#
#
#         print('input ', out_indices.size()[0], ' index tuples')
#
#         super().__init__(in_rank=3, out_size=out_size, temp_indices=temp_indices, additional=additional, subsample=subsample)
#
#         assert(len(in_size) == 3)
#
#         self.in_size = in_size
#         self.k = k
#         self.sigma_scale = sigma_scale
#         self.num_values = num_values
#         self.min_sigma = min_sigma
#         self.out_size = out_size
#         self.adaptive = adaptive
#         self.pre = pre
#         self.mix = mix
#
#         if mix:
#             assert(pre == out_size[0])
#             self.alpha = Parameter(torch.randn(1))
#
#         # outsize = k * prod(out_size) * 5
#
#         # one-hot matrix for the inputs to the hypernetwork
#         one_hots = torch.zeros(out_indices.size()[0], sum(out_size) + k)
#         for r in range(out_indices.size()[0]):
#
#             min = 0
#             for i in range(len(out_size)):
#                 one_hots[r, min + int(out_indices[r, i])] = 1
#                 min += out_size[i]
#
#             one_hots[r, min + r % k] = 1
#             # print(out_indices[r, :], out_size)
#             # print(one_hots[r, :])
#
#         # convert out_indices to float values that return the correct indices when sigmoided.
#         # out_indices = inv(out_indices, torch.FloatTensor(out_size).unsqueeze(0).expand_as(out_indices))
#         self.register_buffer('one_hots', one_hots)
#
#         if self.adaptive:
#             activation = nn.ReLU()
#
#             assert(pre > 0)
#
#             p1 = 4
#             p2 = 2
#
#             c , w, h = in_size
#             hid = max(1, floor(floor(w/p1)/p2) * floor(floor(h/p1)/p2)) * 32
#
#             self.preprocess = nn.Sequential(
#                 #nn.MaxPool2d(kernel_size=4),
#                 # util.Debug(lambda x: print(x.size())),
#                 nn.Conv2d(c, 4, kernel_size=5, padding=2),
#                 activation,
#                 nn.Conv2d(4, 4, kernel_size=5, padding=2),
#                 activation,
#                 nn.MaxPool2d(kernel_size=p1),
#                 nn.Conv2d(4, 16, kernel_size=5, padding=2),
#                 activation,
#                 nn.Conv2d(16, 16, kernel_size=5, padding=2),
#                 activation,
#                 nn.MaxPool2d(kernel_size=p2),
#                 nn.Conv2d(16, 32, kernel_size=5, padding=2),
#                 activation,
#                 nn.Conv2d(32, 32, kernel_size=5, padding=2),
#                 activation,
#                 # util.Debug(lambda x : print(x.size())),
#                 util.Flatten(),
#                 nn.Linear(hid, 64),
#                 nn.Dropout(DROPOUT),
#                 activation,
#                 nn.Linear(64, 64),
#                 nn.Dropout(DROPOUT),
#                 activation,
#                 nn.Linear(64, pre),
#                 nn.Sigmoid()
#             )
#
#             hidden = 64
#             self.source = nn.Sequential(
#                 nn.Linear(pre + sum(out_size) + k, hidden), # input + output index (one hots) + k (one hot)
#                 activation,
#                 nn.Linear(hidden, hidden),
#                 activation,
#                 nn.Linear(hidden, hidden),
#                 activation,
#                 nn.Linear(hidden, 4),
#             )
#
#             self.sigmas = Parameter(torch.randn((1, self.k * prod(out_size), 1)))
#
#         else:
#             self.nas = Parameter(torch.randn((self.k * prod(out_size), 5)))
#
#         self.bias = Parameter(torch.zeros(*out_size))
#
#         if num_values > 0:
#             self.values = Parameter(torch.randn((num_values,)))
#
#     def hyper(self, input):
#         """
#         Evaluates hypernetwork.
#         """
#
#         b, c, w, h = input.size()
#         # l, d  = self.out_indices.size() # prod(out_shape) * k
#         l, dh = self.one_hots.size()
#
#         # outs = Variable(self.out_indices.unsqueeze(0).expand(b, l, d))
#         hots = Variable(self.one_hots.unsqueeze(0).expand(b, l, dh))
#
#         if self.adaptive:
#
#             input = self.preprocess(input)
#
#             b, d = input.size()
#             assert(d == self.pre)
#
#             input = input.unsqueeze(1).expand(b, l, d)
#             input = torch.cat([input, hots], dim=2)
#
#             input = input.view(b*l, -1)
#
#             res = self.source(input).view(b, l , 4)
#
#             ss = self.sigmas.expand(b, l, 1)
#
#             res = torch.cat([res[:, :, :-1], ss, res[:, :, -1:]], dim=2)
#
#         else:
#             res = self.nas.unsqueeze(0).expand(b, l, 5)
#
#         means, sigmas, values = self.split_out(res, self.in_size)
#
#         sigmas = sigmas * self.sigma_scale + self.min_sigma
#
#         if self.num_values > 0:
#             mult = l // self.num_values
#
#             values = self.values.unsqueeze(0).expand(mult, self.num_values)
#             values = values.contiguous().view(-1)[:l]
#
#             values = values.unsqueeze(0).expand(b, l)
#
#         self.last_values = values.data
#
#         return means, sigmas, values, self.bias
#
#     def forward(self, input):
#
#         self.last_out = super().forward(input)
#
#         return self.last_out
#
#     def plot(self, images):
#         perrow = 5
#
#         num, c, w, h = images.size()
#
#         rows = int(math.ceil(num/perrow))
#
#         means, sigmas, values, _ = self.hyper(images)
#
#         images = images.data
#
#         plt.figure(figsize=(perrow * 3, rows*3))
#
#         for i in range(num):
#
#             ax = plt.subplot(rows, perrow, i+1)
#
#             im = np.transpose(images[i, :, :, :].cpu().numpy(), (1, 2, 0))
#             im = np.squeeze(im)
#
#             ax.imshow(im, interpolation='nearest', extent=(-0.5, w-0.5, -0.5, h-0.5), cmap='gray_r')
#
#             util.plot(means[i, :, 1:].unsqueeze(0), sigmas[i, :, 1:].unsqueeze(0), values[i, :].unsqueeze(0), axes=ax)
#
#         plt.gcf()

def rescale( image, outsize):
    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))

    image = resize(image, outsize)

    return torch.from_numpy( np.transpose(image, (2, 0, 1)) )

class SimpleImageLayer(gaussian_temp.HyperLayer):
    """
    NB: k is the number of tuples per input dimension. That is, k = 4 results in a 16 * c grid of inputs evenly spaced
     across a bounding box
    """

    def __init__(self, in_size, k, adaptive=True, additional=0, sigma_scale=0.1, num_values=-1, min_sigma=0.0, subsample=None, preprocess=None):


        ci, hi, wi = in_size
        out_size = co, ho, wo = ci, k, k

        num_indices = k * k * ci * co;

        indices = torch.LongTensor(list(np.ndindex( (k, k, co) )))[:, (2, 0, 1)]

        pixel_indices  = indices[:, 1:3].clone()

        indices = torch.cat([
            indices,
            indices[:, 0:1],
            indices[:, 1:3].clone().fill_(0.0)
            ], dim=1)

        self.lc = [4, 5]
        self.lc_sizes = [(out_size+in_size)[i] for i in self.lc]

        super().__init__(in_rank=3, out_size=(co, ho, wo), temp_indices=indices, learn_cols=self.lc, additional=additional, subsample=subsample)

        # scale to [0,1] in each dim
        pixel_indices = pixel_indices.float() / torch.FloatTensor([k, k]).unsqueeze(0).expand_as(pixel_indices)

        self.register_buffer('pixel_indices', pixel_indices)

        assert(len(in_size) == 3)

        self.in_size = in_size
        self.k = k
        self.sigma_scale = sigma_scale
        self.num_values = num_values
        self.min_sigma = min_sigma
        self.out_size = out_size
        self.adaptive = adaptive

        if self.adaptive:
            activation = nn.ReLU()

            p1 = 4
            p2 = 2

            c , w, h = in_size

            if preprocess is not None:
                self.preprocess = preprocess
            else:
                # default preprocess
                hid = max(1, floor(floor(w/p1)/p2) * floor(floor(h/p1)/p2)) * 32

                self.preprocess = nn.Sequential(
                    #nn.MaxPool2d(kernel_size=4),
                    # util.Debug(lambda x: print(x.size())),
                    nn.Conv2d(c, 4, kernel_size=5, padding=2),
                    activation,
                    nn.Conv2d(4, 4, kernel_size=5, padding=2),
                    activation,
                    nn.MaxPool2d(kernel_size=p1),
                    nn.Conv2d(4, 16, kernel_size=5, padding=2),
                    activation,
                    nn.Conv2d(16, 16, kernel_size=5, padding=2),
                    activation,
                    nn.MaxPool2d(kernel_size=p2),
                    nn.Conv2d(16, 32, kernel_size=5, padding=2),
                    activation,
                    nn.Conv2d(32, 32, kernel_size=5, padding=2),
                    activation,
                    # util.Debug(lambda x : print(x.size())),
                    util.Flatten(),
                    nn.Linear(hid, 64),
                    nn.Dropout(DROPOUT),
                    activation,
                    nn.Linear(64, 64),
                    nn.Dropout(DROPOUT),
                    activation,
                    nn.Linear(64, 4),
                )

            # else:  # Use a small convnet to select the bounding box
            #     hid = max(1, floor(w/5) * floor(h/5) * c)
            #     self.preprocess = nn.Sequential(
            #         nn.Conv2d(c, c, kernel_size=5, padding=2),
            #         activation,
            #         nn.Conv2d(c, c, kernel_size=5, padding=2),
            #         activation,
            #         nn.Conv2d(c, c, kernel_size=5, padding=2),
            #         activation,
            #         nn.MaxPool2d(kernel_size=5),
            #         util.Flatten(),
            #         nn.Linear(hid, 16),
            #         activation,
            #         nn.Linear(16, 4)
            #     )

            self.register_buffer('bbox_offset', torch.FloatTensor([-1, 1, -1, 1]))
            # self.register_buffer('bbox_offset', torch.FloatTensor([0, 5, 1, 2]))

        else: # if not adaptive
            self.bound = Parameter(torch.FloatTensor([-1, 1, -1, 1]))

        self.sigmas = Parameter(torch.randn( (indices.size(0), ) ))

        if num_values > 0:
            self.values = Parameter(torch.randn((num_values,)))
        else:
            self.values = Parameter(torch.randn( (indices.size(0), ) ))

        self.bias = Parameter(torch.zeros(*self.out_size))

    def hyper(self, input, prep=None):
        """
        Evaluates hypernetwork.
        """
        # print('!!!', prep.size())

        b, c, h, w = input.size()
        l = self.pixel_indices.size(0)

        if self.adaptive:

            if prep is None:
                bbox = self.preprocess(input)
            else:
                bbox = prep

            # ensure that the bounding box covers a reasonable area of the image at the start
            bbox = bbox + self.bbox_offset

        else:
            bbox = self.bound.unsqueeze(0).expand(b, 4)

        ymin, ymax, xmin, xmax = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3] # y is vert (height), x is hor (width),

        yrange, xrange = ymax - ymin, xmax - xmin

        pih, piw = self.pixel_indices.size()
        pixel_indices = self.pixel_indices.unsqueeze(0).expand(b, pih, piw)

        range = torch.cat([yrange.unsqueeze(1), xrange.unsqueeze(1)], dim=1).unsqueeze(1)
        range = range.expand_as(pixel_indices)

        min = torch.cat([ymin.unsqueeze(1), xmin.unsqueeze(1)], dim=1).unsqueeze(1)
        min = min.expand_as(pixel_indices)

        pixel_scaled = pixel_indices * range + min

        # Expand to batch dim
        sigmas = self.sigmas.unsqueeze(0).expand(b, l)

        # Unpack the values
        if self.num_values > 0:
            mult = l // self.num_values

            values = self.values.unsqueeze(0).expand(mult, self.num_values)
            values = values.contiguous().view(-1)[:l]

            # Expand to batch dim
            values = values.unsqueeze(0).expand(b, l)
        else:
            values = self.values.unsqueeze(0).expand(b, l)

        res = torch.cat([pixel_scaled, sigmas.unsqueeze(2), values.unsqueeze(2)], dim=2)

        means, sigmas, values = self.split_out(res, self.lc_sizes)

        sigmas = sigmas * self.sigma_scale + self.min_sigma

        self.last_values = values.data

        return means, sigmas, values, self.bias

    def forward(self, input, prep=None):

        self.last_out = super().forward(input, prep=prep)

        return self.last_out

    def plot(self, images):
        perrow = 5

        num, c, w, h = images.size()

        rows = int(math.ceil(num/perrow))

        means, sigmas, values, _ = self.hyper(images)

        images = images.data

        plt.figure(figsize=(perrow * 3, rows*3))

        for i in range(num):

            ax = plt.subplot(rows, perrow, i+1)

            im = np.transpose(images[i, :, :, :].cpu().numpy(), (1, 2, 0))
            im = np.squeeze(im)

            ax.imshow(im, interpolation='nearest', extent=(-0.5, w-0.5, -0.5, h-0.5), cmap='gray_r')

            util.plot(means[i, :, 1:].unsqueeze(0), sigmas[i, :, 1:].unsqueeze(0), values[i, :].unsqueeze(0), axes=ax, flip_y=h, alpha_global=0.2)

        plt.gcf()

class ASHModel(nn.Module):

    def __init__(self, shape, k, glimpses, additional, num_values, min_sigma, subsample, hidden, num_classes, reinforce=False):
        super().__init__()

        self.reinforce = reinforce

        activation = nn.ReLU()

        p1 = 4
        p2 = 2

        ch1, ch2, ch3 = 64, 128, 512

        c, h, w = shape
        hid = max(1, floor(floor(w / p1) / p2) * floor(floor(h / p1) / p2)) * ch3
        hidlin = 512

        print(hid)

        self.preprocess = nn.Sequential(
            # nn.MaxPool2d(kernel_size=4),
            # util.Debug(lambda x: print(x.size())),
            nn.Conv2d(c, ch1, kernel_size=3, padding=1),
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
            #util.Debug(lambda x : print(x.size())),
            util.Flatten(),
            nn.Linear(hid, hidlin),
            nn.Dropout(DROPOUT),
            activation,
            nn.Linear(hidlin, hidlin),
            nn.Dropout(DROPOUT),
            activation,
            nn.Linear(hidlin, 4 if not self.reinforce else 8)
        )

        # hid = max(1, floor(w / 5) * floor(h / 5) * c)
        # self.preprocess = nn.Sequential(
        #     nn.Conv2d(c, c, kernel_size=5, padding=2),
        #     activation,
        #     nn.Conv2d(c, c, kernel_size=5, padding=2),
        #     activation,
        #     nn.Conv2d(c, c, kernel_size=5, padding=2),
        #     activation,
        #     nn.MaxPool2d(kernel_size=5),
        #     util.Flatten(),
        #     nn.Linear(hid, 16),
        #     activation,
        #     nn.Linear(16, 4*glimpses)
        # )

        self.hyperlayers = []

        for _ in range(glimpses):
            self.hyperlayers.append(SimpleImageLayer(shape, k=k, adaptive=True, additional=additional,
                                             num_values=num_values,
                                             min_sigma=min_sigma, subsample=subsample))

        self.lin1 = nn.Linear(k * k * shape[0] * glimpses, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)

        self.k = k
        self.is_cuda = False

    def cuda(self):

        super().cuda()
        self.is_cuda = True

        for hyper in self.hyperlayers:
            hyper.apply(lambda t: t.cuda())

    def forward(self, image):

        prep = self.preprocess(image)

        b = image.size(0)
        glimpses = []

        if not self.reinforce:

            for i, hyper in enumerate(self.hyperlayers):
                glimpses.append(hyper(image, prep=prep[:, i*4 : (i+1)*4]))

            x = torch.cat(glimpses, dim=1).view(b, -1)
            x = F.relu(self.lin1(x))
            x = F.softmax(self.lin2(x), dim=1)

            return x

        else:
            # RL baseline
            stoch_nodes = []
            samples = []
            for i, hyper in enumerate(self.hyperlayers):
                ps = prep[:, i * 8: (i + 1) * 8]
                bbox  = ps[:, :4]
                sigs  = ps[:, 4:]

                stoch_node = torch.distributions.Normal(0, F.softplus(sigs))
                sample = stoch_node.sample()

                stoch_nodes.append(stoch_node)
                samples.append(sample)

                bbox = bbox + sample

                glimpses.append(self.extract(image, bbox, (self.k, self.k)))

            x = torch.cat(glimpses, dim=1).view(b, -1)
            x = F.relu(self.lin1(x))
            x = F.softmax(self.lin2(x), dim=1)

            return x, stoch_nodes, samples

    def extract(self, image, bbox, res=(12,12)):
        """
        Extracts a rectangle from the given image and scales it down.

        :param image:
        :param bbox:
        :param res:
        :return:
        """

        b, c, h, w  = image.size()

        bbox = F.sigmoid(bbox)
        bbox[:, :2] = (bbox[:, :2] - gaussian.EPSILON) * h
        bbox[:, 2:] = (bbox[:, 2:] - gaussian.EPSILON) * w
        bbox = bbox.round().long()

        y, x = bbox[:, :2], bbox[:, 2:]  # y is vert (height), x is hor (width),

        # flip the bounds that are the wrong way around
        y, _ = torch.sort(y, dim=1)
        x, _ = torch.sort(x, dim=1)

        extract = torch.cuda.FloatTensor(b, c, res[0], res[1]).zero_() if self.is_cuda else torch.FloatTensor(b, c, res[0], res[1]).zero_()

        for bi in range(b):
            if (x[bi, 0] - x[bi, 1]).abs() > 2 and (y[bi, 0] - y[bi, 1]).abs() > 2:

                temp = image[bi, :, y[bi, 0]:y[bi, 1], x[bi, 0]:x[bi, 1],]

                extract[bi] = rescale(temp, outsize=res)

        extract = Variable(extract)

        return extract


    def debug(self):
        print(list(self.preprocess.parameters())[0].grad)

    def plot(self, images):

        prep = self.preprocess(images)

        perrow = 5

        num, c, w, h = images.size()

        rows = int(math.ceil(num / perrow))

        plt.figure(figsize=(perrow * 3, rows * 3))

        for i in range(num):
            ax = plt.subplot(rows, perrow, i + 1)

            im = np.transpose(images.data[i, :, :, :].cpu().numpy(), (1, 2, 0))
            im = np.squeeze(im)

            ax.imshow(im, interpolation='nearest', extent=(-0.5, w - 0.5, -0.5, h - 0.5), origin='upper', cmap='gray_r')

            for j, hyper in enumerate(self.hyperlayers):
                means, sigmas, values, _ = hyper.hyper( images, prep=prep[:, j*4 : (j+1)*4] )

                util.plot(means[i, :].unsqueeze(0), sigmas[i, :].unsqueeze(0), values[i, :].unsqueeze(0),
                    axes=ax, flip_y=h, alpha_global=0.3)

            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)

            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(-0.5, h - 0.5) # NB Axis flipped

        plt.gcf()

# class ToImageLayer(gaussian_out.HyperLayer):
#     """
#     Simple hyperlayer for the 1D MNIST experiment
#
#     NB: k is the number of tuples _per hidden node_.
#     """
#
#     def __init__(self, in_size, out_size, k, adaptive=True, additional=0, sigma_scale=0.1, num_values=-1, min_sigma=0.0, pre=0, subsample=None):
#
#         in_indices = torch.LongTensor(list(np.ndindex(in_size)))
#
#         in_indices = in_indices.unsqueeze(1).expand(prod(in_size), k, len(in_size))
#         in_indices = in_indices.contiguous().view(prod(in_size) * k, len(in_size))
#
#         print('reconstruction  ',  in_indices.size()[0], ' index tuples')
#
#         super().__init__(in_size=in_size, out_size=out_size, in_indices=in_indices, additional=additional, subsample=subsample)
#
#         assert(len(out_size) == 3)
#
#         self.in_size = in_size
#         self.k = k
#         self.sigma_scale = sigma_scale
#         self.num_values = num_values
#         self.min_sigma = min_sigma
#         self.out_size = out_size
#         self.adaptive = adaptive
#         self.pre = pre
#
#         # outsize = k * prod(out_size) * 5
#
#         # one-hot matrix for the inputs to the hypernetwork
#         one_hots = torch.zeros(in_indices.size()[0], sum(in_size) + k)
#         for r in range(in_indices.size()[0]):
#
#             min = 0
#             for i in range(len(in_size)):
#                 one_hots[r, min + int(in_indices[r, i])] = 1
#                 min += in_size[i]
#
#             one_hots[r, min + r % k] = 1
#             # print(out_indices[r, :], out_size)
#             # print(one_hots[r, :])
#
#         # convert out_indices to float values that return the correct indices when sigmoided.
#         # out_indices = inv(out_indices, torch.FloatTensor(out_size).unsqueeze(0).expand_as(out_indices))
#         self.register_buffer('one_hots', one_hots)
#
#         if self.adaptive:
#             activation = nn.ReLU()
#
#             assert(pre > 0)
#
#             self.preprocess = nn.Sequential(
#                 util.Flatten(),
#                 nn.Linear(prod(in_size), 64),
#                 activation,
#                 nn.Linear(64, 64),
#                 activation,
#                 nn.Linear(64, 64),
#                 nn.Dropout(DROPOUT),
#                 activation,
#                 nn.Linear(64, pre),
#                 nn.Sigmoid()
#             )
#
#             self.source = nn.Sequential(
#                 nn.Linear(pre + sum(in_size) + k, 64), # input + output index (one hots) + k (one hot)
#                 activation,
#                 nn.Linear(64, 64),
#                 activation,
#                 nn.Linear(64, 64),
#                 activation,
#                 nn.Linear(64, 4),
#             )
#
#             self.sigmas = Parameter(torch.randn((1, self.k * prod(in_size), 1)))
#
#         else:
#             self.nas = Parameter(torch.randn((self.k * prod(in_size), 5)))
#
#         self.bias = Parameter(torch.zeros(*out_size))
#
#         if num_values > 0:
#             self.values = Parameter(torch.randn((num_values,)))
#
#     def hyper(self, input):
#         """
#         Evaluates hypernetwork.
#         """
#
#         b = input.size()[0]
#         l, dh = self.one_hots.size()
#
#         hots = Variable(self.one_hots.unsqueeze(0).expand(b, l, dh))
#
#         if self.adaptive:
#
#             input = self.preprocess(input)
#
#             b, d = input.size()
#             assert(d == self.pre)
#
#             input = input.unsqueeze(1).expand(b, l, d)
#             input = torch.cat([input, hots], dim=2)
#
#             input = input.view(b*l, -1)
#
#             res = self.source(input).view(b, l , 4)
#
#             ss = self.sigmas.expand(b, l, 1)
#
#             res = torch.cat([res[:, :, :-1], ss, res[:, :, -1:]], dim=2)
#
#         else:
#             res = self.nas.unsqueeze(0).expand(b, l, 5)
#
#         means, sigmas, values = self.split_out(res, self.out_size)
#
#         sigmas = sigmas * self.sigma_scale + self.min_sigma
#
#         if self.num_values > 0:
#             mult = l // self.num_values
#
#             values = self.values.unsqueeze(0).expand(mult, self.num_values)
#             values = values.contiguous().view(-1)[:l]
#
#             values = values.unsqueeze(0).expand(b, l)
#
#         self.last_values = values.data
#
#         return means, sigmas, values, self.bias
#
#     def plot(self, input):
#         perrow = 5
#
#         images = self.forward(input)
#         num, c, w, h = images.size()
#
#         rows = int(math.ceil(num/perrow))
#
#         means, sigmas, values, _ = self.hyper(input)
#
#         images = images.data
#
#         plt.figure(figsize=(perrow * 3, rows*3))
#
#         for i in range(num):
#
#             ax = plt.subplot(rows, perrow, i+1)
#
#             im = np.transpose(images[i, :, :, :].cpu().numpy(), (1, 2, 0))
#             im = np.squeeze(im)
#
#             ax.imshow(im, interpolation='nearest', extent=(-0.5, w-0.5, -0.5, h-0.5), cmap='gray_r')
#
#             util.plot(means[i, :, 1:].unsqueeze(0), sigmas[i, :, 1:].unsqueeze(0), values[i, :].unsqueeze(0), axes=ax)
#
#         plt.gcf()

PLOT = True
COLUMN = 13

def go(args, batch=64, epochs=350, k=3, additional=64, modelname='baseline', cuda=False,
       seed=1, lr=0.001, subsample=None, num_values=-1, min_sigma=0.0,
       tb_dir=None, data='./data', hidden=32, task='mnist', final=False, pre=3, dropout=0.0,
       rec_lambda=None, small=True):

    DROPOUT = dropout

    FT = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(seed)

    tbw = SummaryWriter(log_dir=tb_dir)

    normalize = transforms.Compose([transforms.ToTensor()])

    if(task=='mnist'):
        data = data + os.sep + task

        if final:
            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=normalize)
            trainloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=2)

            test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=normalize)
            testloader = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=normalize)

            trainloader = DataLoader(train, batch_size=batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

        shape = (1, 28, 28)
        num_classes = 10

    elif (task == 'image-folder-bw'):

        if final:
            raise Exception('not implemented yet')
        else:

            tr = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.ImageFolder(root=data, transform=tr)

            trainloader = DataLoader(train, batch_size=batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

        shape = (1, 100, 100)
        num_classes = 10

    elif(task=='cifar10'):
        data = data + os.sep + task

        if final:
            train = torchvision.datasets.CIFAR10(root=data, train=True, download=True, transform=normalize)
            trainloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=2)
            test = torchvision.datasets.CIFAR10(root=data, train=False, download=True, transform=normalize)
            testloader = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.CIFAR10(root=data, train=True, download=True, transform=normalize)

            trainloader = DataLoader(train, batch_size=batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))


        shape = (3, 32, 32)
        num_classes = 10

    elif(task=='cifar100'):

        data = data + os.sep + task

        if final:
            train = torchvision.datasets.CIFAR100(root=data, train=True, download=True, transform=normalize)
            trainloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=2)
            test = torchvision.datasets.CIFAR100(root=data, train=False, download=True, transform=normalize)
            testloader = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.CIFAR100(root=data, train=True, download=True, transform=normalize)

            trainloader = DataLoader(train, batch_size=batch, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=batch, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

        shape = (3, 32, 32)
        num_classes = 100

    else:
        raise Exception('Task name {} not recognized'.format(task))

    activation = nn.ReLU()

    hyperlayer = None

    reinforce = False

    if modelname == 'baseline':

        model = nn.Sequential(
            util.Flatten(),
            nn.Linear(prod(shape), hidden),
            activation,
            nn.Linear(hidden, num_classes),
            nn.Softmax())

    elif modelname == 'baseline-conv':

        c, w, h = shape
        hid = floor(floor(w / 8) / 4) * floor(floor(h / 8) / 4) * 32

        model = nn.Sequential(
            nn.Conv2d(c, 4, kernel_size=5, padding=2),
            activation,
            nn.Conv2d(4, 4, kernel_size=5, padding=2),
            activation,
            nn.MaxPool2d(kernel_size=8),
            nn.Conv2d(4, 16, kernel_size=5, padding=2),
            activation,
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            activation,
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            activation,
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            activation,
            util.Flatten(),
            nn.Linear(hid, 128),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
            nn.Softmax()
        )

    elif modelname == 'ash':

        model = ASHModel(shape, k, args.num_glimpses, additional, num_values, min_sigma, subsample, hidden, num_classes, reinforce=False)

        # model = nn.Sequential(
        #     hyperlayer,
        #     util.Flatten(),
        #     nn.Linear(k*k*C, hidden),
        #     activation,
        #     nn.Linear(hidden, num_classes),
        #     nn.Softmax())

    elif modelname == 'ash-reinforce':

        model = ASHModel(shape, k, args.num_glimpses, additional, num_values, min_sigma, subsample, hidden, num_classes, reinforce=True)
        reinforce = True

    elif modelname == 'nas':
        C = 1
        hyperlayer = SimpleImageLayer(shape, out_channels=C, k=k, adaptive=False, additional=additional, num_values=num_values,
                                min_sigma=min_sigma, subsample=subsample)
        #
        # if rec_lambda is not None:
        #     reconstruction = ToImageLayer((C, k, k), out_size=shape, k=k, adaptive=False, additional=additional, num_values=num_values,
        #                         min_sigma=min_sigma, subsample=subsample, pre=pre)

        model = nn.Sequential(
            hyperlayer,
            util.Flatten(),
            nn.Linear(k*k*C, hidden),
            activation,
            nn.Linear(hidden, num_classes),
            nn.Softmax())

    # elif modelname == 'ash-conv':
    #     C = 1
    #     hyperlayer = SimpleImageLayer(shape, out_channels=C, k=k, adaptive=True, additional=additional, num_values=num_values,
    #                             min_sigma=min_sigma, subsample=subsample, big=not small)
    #
    #     model = nn.Sequential(
    #         hyperlayer,
    #         activation,
    #         nn.Conv2d(C, 16, kernel_size=5, padding=2), activation,
    #         nn.Conv2d(16, 16, kernel_size=5, padding=2), activation,
    #         nn.Conv2d(16, 16, kernel_size=5, padding=2), activation,
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Conv2d(16, 32, kernel_size=5, padding=2), activation,
    #         nn.Conv2d(32, 32, kernel_size=5, padding=2), activation,
    #         nn.Conv2d(32, 32, kernel_size=5, padding=2), activation,
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Conv2d(32, 64, kernel_size=5, padding=2), activation,
    #         nn.Conv2d(64, 64, kernel_size=5, padding=2), activation,
    #         nn.Conv2d(64, 64, kernel_size=5, padding=2), activation,
    #         nn.MaxPool2d(kernel_size=2),
    #         nn.Conv2d(64, 128, kernel_size=5, padding=2), activation,
    #         nn.MaxPool2d(kernel_size=2),
    #         util.Flatten(),
    #         nn.Linear(128, num_classes),
    #         nn.Softmax())

    else:
        raise Exception('Model name {} not recognized'.format(modelname))

    if cuda:
        model.cuda()
        if hyperlayer is not None:
            hyperlayer.apply(lambda t: t.cuda())
        # if rec_lambda is not None:
        #     reconstruction.apply(lambda t: t.cuda())

    # if rec_lambda is None:
    #     optimizer = optim.Adam(model.parameters(), lr=lr)
    # else:
    #     optimizer = optim.Adam(list(model.parameters()) + list(reconstruction.parameters()), lr=lr)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    xent = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    step = 0

    sigs, vals = [], []

    util.makedirs('./mnist/')

    for epoch in range(epochs):

        model.train()

        for i, data in tqdm(enumerate(trainloader, 0)):

            # get the inputs
            inputs, labels = data

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            if not reinforce:
                outputs = model(inputs)
            else:
                outputs, stoch_nodes, actions = model(inputs)

            mloss = F.cross_entropy(outputs, labels, reduce=False)

            if reinforce:
                rloss = 0.0

                for node, action in zip(stoch_nodes, actions):
                    rloss = rloss - node.log_prob(action) * - mloss.data.unsqueeze(1).expand_as(action)

                # print(mloss.size(), rloss.size())

                loss = rloss.sum(dim=1) + mloss
            else:
                loss = mloss

            loss = loss.sum()
            loss.backward()  # compute the gradients

            # model.debug()

            # print(hyperlayer.values, hyperlayer.values.grad)

            optimizer.step()

            tbw.add_scalar('mnist/train-loss', float(loss.data.item()), step)

            step += inputs.size(0)

            if PLOT and i == 0 and hyperlayer is not None:

                sigmas = list(hyperlayer.last_sigmas[0, :])
                values = list(hyperlayer.last_values[0, :])

                sigs.append(sigmas)
                vals.append(values)

                ax = plt.figure().add_subplot(111)

                for j, (s, v) in enumerate(zip(sigs, vals)):
                    s = [si.item() for si in s]
                    ax.scatter([j] * len(s), s, c=v, linewidth=0,  alpha=0.2, cmap='RdYlBu', vmin=-1.0, vmax=1.0)

                ax.set_aspect('auto')
                plt.ylim(ymin=0)
                util.clean()

                plt.savefig('sigmas.pdf')
                plt.savefig('sigmas.png')

                hyperlayer.plot(inputs[:10, ...])
                plt.savefig('mnist/attention.{:03}.pdf'.format(epoch))

            if PLOT and i == 0 and type(model) is ASHModel and not reinforce:

                model.plot(inputs[:10, ...])
                plt.savefig('mnist/attention.glimpses.{:03}.pdf'.format(epoch))

        total = 0.0
        correct = 0.0

        model.eval()

        for i, data in enumerate(testloader, 0):

            # get the inputs
            inputs, labels = data

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)

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

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Which model to train.",
                        default='baseline')

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples in the decoder layer",
                        default=3, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled",
                        default=64, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-Z", "--small-hyper",
                        dest="small",
                        help="Whether to use a small hypernet.",
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

    parser.add_argument("-F", "--num-values", dest="num_values",
                        help="How many fixed values to allow the network",
                        default=-1, type=int)

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

    parser.add_argument("-p", "--pre", dest="pre",
                        help="Size of the preprocessed input representation.",
                        default=32, type=int)

    parser.add_argument("-Q", "--dropout", dest="dropout",
                        help="Dropout of the baseline and hypernetwork.",
                        default=0.0, type=float)

    parser.add_argument("-R", "--reconstruction-loss", dest="rec_loss",
                        help="Reconstruction loss parameter.",
                        default=None, type=float)

    parser.add_argument("-G", "--num-glimpses", dest="num_glimpses",
                        help="Number of glimpses for the ash model.",
                        default=4, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(epochs=options.epochs, batch=options.batch_size, k=options.k,
       additional=options.additional, modelname=options.model, cuda=options.cuda,
       lr=options.lr, subsample=options.subsample,
       num_values=options.num_values, min_sigma=options.min_sigma,
       tb_dir=options.tb_dir, data=options.data, task=options.task,
       final=options.final, hidden=options.hidden, pre=options.pre,
       dropout=options.dropout, rec_lambda=options.rec_loss,
       small=options.small, seed=options.seed, args=options)
