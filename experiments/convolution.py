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


### Wide resnet. Ported from fast.ai lib

def _bn(ni, init_zero=False):
    """
    Batchnorm layer with (optional) 0 initialization

    :param ni:
    :param init_zero:
    :return:
    """
    m = nn.BatchNorm2d(ni)
    m.weight.data.fill_(0 if init_zero else 1)
    m.bias.data.zero_()
    return m

def bn_relu_conv(ni, nf, ks, stride, init_zero=False):
    """
    Batchnorm/relu/convolution block
    """

    bn_initzero = _bn(ni, init_zero=init_zero)

    return nn.Sequential(
        bn_initzero,
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=ks, stride=stride, padding=ks//2)
    )


class WideBlock(nn.Module):
    """
    Wide ResNet block
    """
    def __init__(self, ni, nf, stride, drop_p=0.0, mult=0.2):
        super().__init__()

        self.mult = mult
        self.bn = nn.BatchNorm2d(ni)

        self.conv1 = nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=3, stride=stride, padding=1)
        self.conv2 = bn_relu_conv(nf, nf, 3, 1)

        self.drop = nn.Dropout(drop_p) if drop_p else None
        self.shortcut = nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=1, stride=stride) if ni != nf else util.Lambda(lambda x : x)

    def forward(self, x):

        x2 = F.relu(self.bn(x))

        r = self.shortcut(x2)
        x = self.conv1(x2)

        if self.drop:
            x = self.drop(x)
        x = self.conv2(x) * self.mult

        return x + r

def _make_group(N, ni, nf, block, stride, drop_p):
    return [block(ni if i == 0 else nf, nf, stride if i == 0 else 1, drop_p) for i in range(N)]

class WideResNet(nn.Module):
    """
    Wide ResNet with `num_groups` and a width of `k`.
    """
    def __init__(self, num_classes:int, num_groups:int=3, N:int=3, k:int=6, drop_p:float=0.0, start_nf:int=16, n_in_channels:int=3, **kwargs):
        super().__init__()

        n_channels = [start_nf]
        for i in range(num_groups): n_channels.append(start_nf*(2**i)*k)

        layers = [nn.Conv2d(n_in_channels, n_channels[0], 3, 1, padding=1)]  # conv1
        for i in range(num_groups):
            layers += _make_group(N, n_channels[i], n_channels[i+1], WideBlock, (1 if i==0 else 2), drop_p)

        layers += [
            nn.BatchNorm2d(n_channels[num_groups]), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), util.Flatten(),
            nn.Linear(n_channels[num_groups], num_classes)
        ]

        self.features = nn.Sequential(*layers)

    def forward(self, x): return self.features(x)

class Cutout(object):
    """
    Cutout transform. From https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py

    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img



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
    def __init__(self, cin, cout, k, gadditional, radditional, region, min_sigma=0.05, sigma_scale=0.05,
                 mmult=0.1, admode='full', edges='sigmoid', bias=True, constrained=False):
        """
        :param k: Number of connections to the input in total
        :param gadditional:
        :param radditional:
        :param pooling factor: The output resolution is reduced by this factor.
        :param region:
        """

        super().__init__()

        # cin, _, _ = in_size
        # cout, hout, wout = cout, hin, win

        self.gadditional, self.radditional = gadditional, radditional

        self.region = (region, region)

        self.min_sigma, self.sigma_scale = min_sigma, sigma_scale
        self.admode, self.mmult = admode, mmult
        self.edges = edges

        if admode == 'none':
            self.mmult *= 0.1
        if edges == 'modulo' or edges == 'clamp':
            self.mmult *= 5.0

        self.k = k
        self.unify = nn.Linear(k*cin, cout, bias=bias)

        if admode == 'none':
            self.params = nn.Parameter(torch.randn(size=(4 if constrained else k * 3, )))
        else:
            if admode == 'full':
                adin = 2 + cin
            elif admode == 'coords':
                adin = 2
            elif admode == 'inputs':
                adin = cin
            else:
                raise Exception(f'adaptivity mode {admode} not recognized')

            # network that generates the coordinates and sigmas
            hidden = cin * 4
            self.toparams = nn.Sequential(
                nn.Linear(adin, hidden), nn.ReLU(),
                nn.Linear(hidden, 4 if constrained else k * 3) # two means, one sigma
            )

        self.register_buffer('mvalues', torch.ones((k,)))

        self.smp = True

        self.constrained = constrained
        if self.constrained:
            kp = int(math.sqrt(k))
            self.register_buffer('base', util.coordinates((kp, kp)) - 0.5)

    def sample(self, smp):
        self.smp = smp

    def hyper(self, x):

        b, c, h, w = x.size()
        k = self.k

        coords = util.coordinates((h, w), cuda=x.is_cuda)

        # the coordinates of the current pixels in parameters space
        # - the index tuples are described relative to these
        hw = torch.tensor((h, w), device=d(x), dtype=torch.float)
        mids = coords[None, :, :, :].expand(b, 2, h, w) * (hw - 1)[None, :, None, None]
        mids = mids.permute(0, 2, 3, 1)
        if self.edges == 'sigmoid':
            mids = util.inv(mids, mx=hw[None, None, None, :])
        mids = mids[:, :, :, None, :].expand(b, h, w, k, 2)

        # add coords to channels
        if self.admode == 'none':
            params = self.params[None, None, None, :].expand(b, h, w, 4 if self.constrained else k * 3)
        else:
            if self.admode == 'full':
                crds = coords[None, :, :, :].expand(b, 2, h, w)
                x = torch.cat([x, crds], dim=1)
            elif self.admode == 'coords':
                x = coords[None, :, :, :].expand(b, 2, h, w)
            elif self.admode == 'inputs':
                pass
            else:
                raise Exception(f'adaptivity mode {self.admode} not recognized')

            x = x.permute(0, 2, 3, 1)
            params = self.toparams(x)

        if self.constrained:
            # Take the three output values (per pixel) and use them to transform a grid of
            # inputs
            MULT = 10.0

            kp = int(math.sqrt(k))

            assert params.size() == (b, h, w, 4)

            scalex = params[:, :, :, 0:1, None, None]
            scaley = params[:, :, :, 1:2, None, None]
            rotate = params[:, :, :, 2:3, None, None]
            sigmas = params[:, :, :, 3:4, None, None]

            coords = self.base.view(1, 1, 1, 2, kp, kp).expand(b, h, w, 2, kp, kp).contiguous() * MULT

            # scale
            coords[:, :, :, 0:1, :, :] *= scalex
            coords[:, :, :, 1:2, :, :] *= scaley

            # rotate
            rotx = coords[:, :, :, 0:1, :, :] * rotate.cos() - coords[:, :, :, 1:2, :, :] * rotate.sin()
            roty = coords[:, :, :, 0:1, :, :] * rotate.sin() + coords[:, :, :, 1:2, :, :] * rotate.cos()

            sigmas = sigmas.expand(b, h, w, 1, kp, kp)
            params = torch.cat([rotx, roty, sigmas], dim=3)

            assert params.size() == (b, h, w, 3, kp, kp)

            params = params.reshape(b, h, w, 3, k).permute(0, 1, 2, 4, 3)
        else:
            params = params.reshape(b, h, w, k, 3)

        assert params.size() == (b, h, w, k, 3) # k index tuples per output pixel

        means  = params[:, :, :, :, :2]
        sigmas = params[:, :, :, :, 2]
        values = self.mvalues[None, None, None, :].expand(b, h, w, k)

        means = mids + self.mmult * means

        s = (h, w)
        means  = sparse.transform_means(means, s, method=self.edges)
        sigmas = sparse.transform_sigmas(sigmas, s, min_sigma=self.min_sigma) * self.sigma_scale

        return means, sigmas, values

    def forward(self, x):

        b, c, h, w = x.size()
        k = self.k
        s = (h, w)

        assert h > 2 and w > 2, 'Input resolution must be larger than 2x2 for the sparse convolution to work.'

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

            weights = props * weights # TODO mult by one, this can be removed
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

    def plot(self, inputs, numpixels=5, ims=None):

        ims = inputs if ims is None else ims

        b, c, h, w = inputs.size()
        b, cims, hims, wims = ims.size()

        k = self.k

        # choose 5 random pixels, for which we'll plot the input pixels.
        choices = torch.randint(low=0, high=h*w, size=(numpixels,))

        perrow = 5

        rows = int(math.ceil(b/perrow))

        means, sigmas, _ = self.hyper(inputs)

        inputs = inputs.data

        plt.figure(figsize=(perrow * 3, rows*3))

        # scale up to image coordinates
        scale = torch.tensor((hims/h, wims/w), device=d(inputs))
        means = means * scale + (scale/2)

        for current in range(b):

            # select subset of means, sigmas
            smeans = means[current, :, :, :, :].view(h*w, k, 2)
            ssigmas = sigmas[current, :, :, :].view(h*w, k, 2)

            color = (torch.arange(numpixels, dtype=torch.float)[:, None].expand(numpixels, k)/numpixels) * 2.0 - 1.0

            smeans = smeans[choices, :, :]
            ssigmas = ssigmas[choices, :]

            ax = plt.subplot(rows, perrow, current+1)

            im = np.transpose(ims[current, :, :, :].cpu().numpy(), (1, 2, 0))
            im = np.squeeze(im)

            ax.imshow(im, interpolation='nearest', extent=(-0.5, wims-0.5, -0.5, hims-0.5), cmap='gray_r')

            util.plot(smeans.reshape(1, -1, 2), ssigmas.reshape(1, -1, 2), color.reshape(1, -1), axes=ax, flip_y=hims, tanh=False)

            # for i, ch in enumerate(choices):
            #     chh, chw = ch//h, ch % w
            #     chh, chw = chh * scale[0] + scale[0]/2, chw * scale[1] + scale[1]/2
            # 
            #     for ik in range(k):
            #         x, y = smeans[i, ik, :]
            # 
            #         l, ml = math.sqrt((y - chw) ** 2 + (x - hims + chh) ** 2), math.sqrt(hims ** 2 + wims ** 2)
            # 
            #         ax.add_line(mpl.lines.Line2D([y, chw], [hims-x, hims-chh], linestyle='-', alpha=max(0.0, (1.0 - (l/ml)) - 0.5), color='white', lw=0.5))

        plt.gcf()

def conv(insize, cout, kernel_size=3, padding=1, sparse=True, **kwargs):

    if sparse:
        layer = Convolution(insize, cout, **kwargs)
    else:
        layer = nn.Conv2d(insize[0], cout, kernel_size=kernel_size, padding=padding)

    return layer

class DavidLayer(nn.Module):

    def __init__(self, insize, cout, sparse=False, **kwargs):
        c, h, w = insize

        super().__init__()

        self.sparse_layers =[]

        self.block0 = nn.Sequential(
            conv(insize, cout, kernel_size=3, padding=1, bias=False, sparse=sparse, **kwargs),
            nn.BatchNorm2d(cout), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block1 = nn.Sequential(
            conv((cout, h//2, w//2), cout, kernel_size=3, padding=1, bias=False, sparse=sparse, **kwargs),
            nn.BatchNorm2d(cout), nn.ReLU()
        )

        self.block2 = nn.Sequential(
            conv((cout, h//2, w//2), cout, kernel_size=3, padding=1, bias=False, sparse=sparse, **kwargs),
            nn.BatchNorm2d(cout), nn.ReLU()
        )

    def forward(self, x):

        x = self.block0(x)

        return x + self.block2(self.block1(x))

    def plot(self, inp, ims, pref, epoch, c=1):

        sparse = list(self.block0.modules())[1]
        sparse.plot(inp, ims=ims)

        plt.savefig(pref + f'/convolution.{epoch:03}.{c}-block0.pdf')
        plt.gcf().clear()

        inp = self.block0(inp)

        sparse = list(self.block1.modules())[1]
        sparse.plot(inp, ims=ims)

        plt.savefig(pref + f'/convolution.{epoch:03}.{c}-block1.pdf')
        plt.gcf().clear()

        inp = self.block1(inp)

        sparse = list(self.block2.modules())[1]
        sparse.plot(inp, ims=ims)

        plt.savefig(pref + f'/convolution.{epoch:03}.{c}-block2.pdf')
        plt.gcf().clear()

    def sample(self, bl):
        list(self.block0.modules())[1].sample(bl)
        list(self.block1.modules())[1].sample(bl)
        list(self.block2.modules())[1].sample(bl)

class DavidNet(nn.Module):

    def __init__(self, insize, num_classes, mul=1.0, sparse=False, **kwargs):
        super().__init__()
        c, h, w = insize

        self.sparse = sparse

        self.prep = nn.Sequential(
            conv((c, h, w), 64, kernel_size=3, padding=1, bias=False, sparse=sparse,**kwargs),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.layer0 = DavidLayer((64, h, w), 128, sparse=sparse, **kwargs) # one maxpool

        self.mid = nn.Sequential(
            conv((128, h//2, w//2), 256, kernel_size=3, padding=1, bias=False, sparse=sparse, **kwargs),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer1 = DavidLayer((256, h//4, w//4), 512, sparse=sparse, **kwargs) # one maxpool

        self.head = nn.Sequential( # h//8, w//8
            nn.MaxPool2d(kernel_size=4),
            util.Flatten(),
            nn.Linear(512 * h//32 * w//32, num_classes),
            util.Lambda(lambda x: x * mul)
        )

    def forward(self, x):

        x = self.prep(x)
        x = self.layer0(x)
        x = self.mid(x)
        x = self.layer1(x)
        x = self.head(x)

        return x

    def plot(self, x, pref, epoch):

        sparse = list(self.prep.modules())[1]
        sparse.plot(x)

        plt.savefig(pref + f'/convolution.{epoch:03}.0-prep.pdf')
        plt.gcf().clear()

        inp = self.prep(x)

        self.layer0.plot(inp, ims=x, pref=pref, epoch=epoch, c=1)

        inp = self.layer0(inp)

        sparse = list(self.mid.modules())[1]
        sparse.plot(inp, ims=x)

        plt.savefig(pref + f'/convolution.{epoch:03}.2-mid.pdf')
        plt.gcf().clear()

        inp = self.mid(inp)

        self.layer1.plot(inp, ims=x, pref=pref, epoch=epoch, c=3)

    def sample(self, bl):

        if self.sparse:
            list(self.prep.modules())[1].sample(bl)
            self.layer0.sample(bl)

            list(self.mid.modules())[1].sample(bl)
            self.layer1.sample(bl)

class DavidOne(nn.Module):
    """
    DavidNet with one adaptive convolution layer (and normal convolutions everywhere else)

    """

    def __init__(self, insize, num_classes, mul=1.0, **kwargs):
        super().__init__()
        c, h, w = insize

        self.prep = nn.Sequential(
            conv((c, h, w), 64, kernel_size=3, padding=1, bias=False, sparse=False, **kwargs),
            nn.BatchNorm2d(64), nn.ReLU()
        )

        self.layer0 = DavidLayer((64, h, w), 128, sparse=False, **kwargs) # one maxpool

        self.mid = nn.Sequential(
            conv((128, h//2, w//2), 256, kernel_size=3, padding=1, bias=False, sparse=True, **kwargs),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer1 = DavidLayer((256, h//4, w//4), 512, sparse=False, **kwargs) # one maxpool

        self.head = nn.Sequential( # h//8, w//8
            nn.MaxPool2d(kernel_size=4),
            util.Flatten(),
            nn.Linear(512 * h//32 * w//32, num_classes),
            util.Lambda(lambda x: x * mul)
        )

    def forward(self, x):

        x = self.prep(x)
        x = self.layer0(x)
        x = self.mid(x)
        x = self.layer1(x)
        x = self.head(x)

        return x

    def plot(self, x, pref, epoch):

        inp = self.prep(x)
        inp = self.layer0(inp)

        sparse = list(self.mid.modules())[1]
        sparse.plot(inp, ims=x)

        plt.savefig(pref + f'/convolution.{epoch:03}.mid.pdf')
        plt.gcf().clear()

    def sample(self, bl):
        """
        Turn sampling on or off.
        :param bl:
        :return:
        """

        list(self.mid.modules())[1].sample(bl)

class Baseline(nn.Module):
    """
    A baseline model with no pooling.

    """

    def __init__(self, num_classes, depth=3, c=16, sparse=False, in_channels=3, **kwargs):
        super().__init__()


        layers = []
        for i in range(depth):
            layers += [
                self.layer(i=in_channels if i ==0 else c, o=c, sparse=sparse, **kwargs), nn.ReLU()
            ]

        self.layers = nn.Sequential(*layers)
        self.toclasses = nn.Linear(c, num_classes)


    def layer(self, i, o, sparse, **kwargs):

        if sparse:
            return Convolution(cin=i, cout=o, **kwargs)

        return nn.Conv2d(i, o, kernel_size=3, padding=1)

    def forward(self, x):

        # convolutions
        x = self.layers(x)

        # global pool
        x = x.mean(-1).mean(-1)

        return self.toclasses(x)

    def sample(self, bl):
        """
        Turn sampling on or off.
        :param bl:
        :return:
        """
        for layer in self.layers:
            if type(layer) == Convolution:
                layer.sample(bl)

    def plot(self, x, pref, epoch):

        with torch.no_grad():
            ims = x
            i = 0

            for layer in self.layers:

                if type(layer) == Convolution:
                    layer.plot(x, ims=ims)
                    plt.savefig(pref + f'/convolution.{epoch:03}.{i}.pdf')
                    i += 1
                    plt.gcf().clear()

                x = layer(x)


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
            # nn.MaxPool2d(kernel_size=2),  # 2x2
            # ResBlock(64), nn.Conv2d(64, 112, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2),  # 4x4
            # ResBlock(112, kernel=5), nn.Conv2d(112, 192, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2), # 2x2
            # ResBlock(192), nn.Conv2d(192, 320, kernel_size=1),
            # nn.MaxPool2d(kernel_size=2), # 1x1
            util.Flatten(),
            nn.Linear(320 * 4 * 4, num_classes)
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

    def plot(self, x, pref, epoch):

        self.sparse.plot(x)
        plt.savefig(pref + f'/convolution.{epoch:03}.0.pdf')
        plt.gcf().clear()

        inp = F.max_pool2d(F.relu(self.sparse(x)), kernel_size=2)

        self.spars1.plot(inp, ims=x)
        plt.savefig(pref + f'/convolution.{epoch:03}.1.pdf')
        plt.gcf().clear()

        inp = F.max_pool2d(F.relu(self.spars1(inp)), kernel_size=2)
        self.spars2.plot(inp, ims=x)
        plt.savefig(pref + f'/convolution.{epoch:03}.2.pdf')

def go(arg):

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

        if arg.augmentation:
            tfms = transforms.Compose(
                [transforms.RandomCrop(size=shape[1:], padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), Cutout(1, 8)]
            )

        data = arg.data + os.sep + arg.task

        if arg.final:
            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=tfms)
            trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch_size, shuffle=True, num_workers=2)

            test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=ToTensor())
            testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch_size, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=tfms)

            trainloader = DataLoader(train, batch_size=arg.batch_size, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=arg.batch_size, sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

    elif (arg.task == 'cifar10'):

        shape = (3, 32, 32)
        num_classes = 10

        if arg.augmentation:
            tfms = transforms.Compose(
                [transforms.RandomCrop(size=shape[1:], padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), Cutout(1, 8),]
            )

        data = arg.data + os.sep + arg.task

        if arg.final:
            train = torchvision.datasets.CIFAR10(root=data, train=True, download=True, transform=tfms)
            trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch_size, shuffle=True, num_workers=2)
            test = torchvision.datasets.CIFAR10(root=data, train=False, download=True, transform=ToTensor())
            testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch_size, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 45000
            NUM_VAL = 5000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.CIFAR10(root=data, train=True, download=True, transform=tfms)

            trainloader = DataLoader(train, batch_size=arg.batch_size, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=arg.batch_size,
                                    sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))

    elif (arg.task == 'imagenette'):

        shape = (3, None, None)
        num_classes = 10

        if arg.augmentation:
            tfms = transforms.Compose(
                [transforms.RandomCrop(size=shape[1:], padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), Cutout(1, 8),]
            )

        if arg.final:
            train = torchvision.datasets.ImageFolder(root=arg.data+os.sep+'train', transform=tfms)
            trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch_size, shuffle=True, num_workers=2)
            test = torchvision.datasets.ImageFolder(root=arg.data+os.sep+'val', transform=tfms)
            testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch_size, shuffle=False, num_workers=2)

        else:
            NUM_TRAIN = 10_000
            NUM_VAL = 3_000
            total = NUM_TRAIN + NUM_VAL

            train = torchvision.datasets.ImageFolder(root=arg.data+os.sep+'train', transform=tfms)

            trainloader = DataLoader(train, batch_size=arg.batch_size, sampler=util.ChunkSampler(0, NUM_TRAIN, total))
            testloader = DataLoader(train, batch_size=arg.batch_size,
                                    sampler=util.ChunkSampler(NUM_TRAIN, NUM_VAL, total))
    else:
        raise Exception('Task {} not recognized'.format(arg.task))

    # Create model
    if arg.model == 'efficient':
        model = Classifier(shape, num_classes, k=arg.k, gadditional=arg.gadditional, radditional=arg.radditional, region=arg.region,
                       admode=arg.admode, sigma_scale=arg.sigma_scale, edges=arg.edges, constrained=arg.constrained)
        sparse = True
    elif arg.model == 'mini':
        model = MiniClassifier(shape, num_classes, k=arg.k, gadditional=arg.gadditional, radditional=arg.radditional, region=arg.region,
                       admode=arg.admode, sigma_scale=arg.sigma_scale, edges=arg.edges, constrained=arg.constrained)
        sparse = False
    elif arg.model == '3c':
        model = ThreeCClassifier(shape, num_classes, k=arg.k, gadditional=arg.gadditional, radditional=arg.radditional, region=arg.region,
                       admode=arg.admode, sigma_scale=arg.sigma_scale, edges=arg.edges, constrained=arg.constrained)
        sparse = True

    elif arg.model == 'david':
        model = DavidNet(insize=shape, num_classes=num_classes, mul=arg.mul, constrained=arg.constrained)
        sparse = False

    elif arg.model == 'david-sparse':
        model = DavidNet(insize=shape, num_classes=num_classes, mul=arg.mul, sparse=True, k=arg.k, gadditional=arg.gadditional, radditional=arg.radditional, region=arg.region,
                       admode=arg.admode, sigma_scale=arg.sigma_scale, edges=arg.edges, constrained=arg.constrained)
        sparse = True

    elif arg.model == 'david-one':
        model = DavidOne(insize=shape, num_classes=num_classes, mul=arg.mul, k=arg.k, gadditional=arg.gadditional, radditional=arg.radditional, region=arg.region,
                       admode=arg.admode, sigma_scale=arg.sigma_scale, edges=arg.edges, constrained=arg.constrained)
        sparse = True

    elif arg.model == 'baseline':
        model = Baseline(num_classes=num_classes, in_channels=shape[0], k=arg.k, gadditional=arg.gadditional, radditional=arg.radditional, region=arg.region,
                       admode=arg.admode, sigma_scale=arg.sigma_scale, edges=arg.edges, constrained=arg.constrained)
        sparse = False

    elif arg.model == 'baseline-sparse':
        model = Baseline(num_classes=num_classes, in_channels=shape[0], sparse=True, k=arg.k, gadditional=arg.gadditional, radditional=arg.radditional, region=arg.region,
                       admode=arg.admode, sigma_scale=arg.sigma_scale, edges=arg.edges, constrained=arg.constrained)
        sparse = True

    elif arg.model == 'wide':
        model = WideResNet(num_classes, constrained=arg.constrained)
        sparse = False
    else:
        raise Exception(f'Model {arg.model} not recognized')


    if arg.cuda:
        model.cuda()

    if arg.optimizer == 'adam':
        opt = torch.optim.Adam(params=model.parameters(), lr=arg.lr)
    elif arg.optimizer == 'adamw':
        opt = torch.optim.AdamW(params=model.parameters(), weight_decay=arg.wd * arg.batch_size, lr=arg.lr)
    elif arg.optimizer == 'nesterov':
        opt = torch.optim.SGD(params=model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=arg.wd * arg.batch_size, nesterov=True)
    else:
        raise Exception(f'Optimizer {arg.optimizer} not recognized.')

    totalsteps = arg.epochs * len(trainloader)
    if arg.schedule == 'tri':
        mid = totalsteps * (2/5)# peak at about 2/5 of the learning process
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: i/mid if i < mid else (totalsteps - i)/(totalsteps - mid) )
    elif arg.schedule == 'warmup':
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i : min(i/(arg.lr_warmup/arg.batch_size), 1.0) )
    elif arg.schedule == 'flat':
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i : 1.0)
    elif arg.schedule == 'exp':
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i : max(1.05 ** (i - totalsteps), 1e-8) )
    elif arg.schedule == 'lin':
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: i / totalsteps)
    else:
        raise Exception(f'Schedule {arg.schedule} not recognized.')

    # Training loop
    util.makedirs(f'./{arg.task}/')
    seen = 0
    for e in range(arg.epochs):

        model.train(True)

        for i, (inputs, labels) in enumerate(tqdm.tqdm(trainloader, 0)):

            b, c, h, w = inputs.size()
            seen += b

            if sparse:
                model.sample(random.random() < arg.sample_prob) # use sampling only on some proprtion of batches

            if arg.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            opt.zero_grad()

            outputs = model(inputs)

            loss = F.cross_entropy(outputs, labels)

            loss.backward()

            # clip gradients
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()
            sch.step()

            tbw.add_scalar('sparsity/loss', loss.item()/b, seen)
            tbw.add_scalar('sparsity/learning_rate', sch.get_lr()[0], seen)

            if sparse and i == 0 and e % arg.plot_every == 0:
                if arg.model == '3c':
                    model.plot(inputs[:10, ...], pref=f'{arg.task}/', epoch=e)
                elif arg.model == 'david-sparse' or arg.model == 'david-one':
                    model.plot(inputs[:10, ...], pref=f'{arg.task}/', epoch=e)
                elif arg.model == 'baseline-sparse':
                    model.plot(inputs[:10, ...], pref=f'{arg.task}/', epoch=e)
                else:
                    model.sparse.plot(inputs[:10, ...])
                    plt.savefig(f'{arg.task}/convolution.{e:03}.pdf')

        # Compute accuracy on test set
        with torch.no_grad():
            if e % arg.test_every == 0:
                model.train(False)

                if hasattr(model, 'sample'):
                    model.sample(False)

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

                print(f'\nepoch {e}: test {acc:.4}', end='')
                tbw.add_scalar('sparsity/test acc', acc, e)

                total, correct = 0.0, 0.0
                for input, labels in trainloader:
                    opt.zero_grad()

                    if arg.cuda:
                        input, labels = input.cuda(), labels.cuda()
                    input, labels = Variable(input), Variable(labels)

                    output = model(input)

                    outcls = output.argmax(dim=1)

                    total += outcls.size(0)
                    correct += (outcls == labels).sum().item()

                acc = correct / float(total)

                print(f' train {acc:.4}')
                tbw.add_scalar('sparsity/train acc', acc, e)


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
                        default=8, type=int)

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

    parser.add_argument("--plot-every",
                        dest="plot_every",
                        help="How many epochs between plotting the sparse indices.",
                        default=1, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many epochs between testing for accuracy.",
                        default=1, type=int)

    parser.add_argument("--admode", dest="admode",
                        help="What input to condition the sparse convolution on (none, full, coords, inputs)",
                        default='full', type=str)

    parser.add_argument("--edges", dest="edges",
                        help="Which operator to use to fit continuous index tuples to the required range (sigmoid, clamp, modulo)).",
                        default='sigmoid', type=str)

    parser.add_argument("--sample-prob",
                        dest="sample_prob",
                        help="Sample probability (with this probability we sample index tuples).",
                        default=0.5, type=float)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping (negative for no clipping).",
                        default=1.0, type=float)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=100000, type=int)

    parser.add_argument("--optimizer",
                        dest="optimizer",
                        help="Which optimizer to use (adam, adamw, nesterov).",
                        default='adam', type=str)

    parser.add_argument("--weight-decay",
                        dest="wd",
                        help="Base weight-decay (multiplied by batch size).",
                        default=0.0005, type=float)

    parser.add_argument("--augmentation", dest="augmentation",
                        help="Whether to apply data augmentation (random crops and random flips).",
                        action="store_true")

    parser.add_argument("--constrained", dest="constrained",
                        help="Whether to use a constrained convolution (pixels form a grid, but squashed and rotated)",
                        action="store_true")

    parser.add_argument("--schedule", dest="schedule",
                        help="Learning rate schedule (flat, tri, warmup, exp)",
                        default='warmup', type=str)

    parser.add_argument("--mul", dest="mul",
                        help="Logit multiplier.",
                        default=1.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
