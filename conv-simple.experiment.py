import sys

import matplotlib as mpl
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import nn
from torch.autograd import Variable
from tqdm import trange

import gaussian
import util
from util import sparsemm

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser

import networkx as nx

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

"""
Simple Graph convolution experiment. Given a set of random vectors, learn to express each as the sum of some of the
others
"""

def clean(axes=None):

    if axes is None:
        axes = plt.gca()

    [s.set_visible(False) for s in axes.spines.values()]
    axes.tick_params(top=False, bottom=False, left=False, right=False, labelbottom=False, labelleft=False)


def densities(points, means, sigmas):
    """
    Compute the unnormalized PDFs of the points under the given MVNs

    (with sigma a diagonal matrix per MVN)

    :param means:
    :param sigmas:
    :param points:
    :return:
    """

    # n: number of MVNs
    # d: number of points per MVN
    # rank: dim of points

    batchsize, n, rank = points.size()
    batchsize, k, rank = means.size()
    # batchsize, k, rank = sigmas.size()

    points = points.unsqueeze(2).expand(batchsize, n, k, rank)
    means  = means.unsqueeze(1).expand_as(points)
    sigmas = sigmas.unsqueeze(1).expand_as(points)

    sigmas_squared = torch.sqrt(1.0/(gaussian.EPSILON + sigmas))

    points = points - means
    points = points * sigmas_squared

    # Compute dot products for all points
    # -- unroll the batch/n dimensions
    points = points.view(-1, 1, rank)
    # -- dot prod
    products = torch.bmm(points, points.transpose(1,2))
    # -- reconstruct shape
    products = products.view(batchsize, n, k)

    num = torch.exp(- 0.5 * products)

    return num

class MatrixHyperlayer(nn.Module):
    """
    Constrained version of the matrix hyperlayer. Each output get exactly k inputs
    """

    def duplicates(self, tuples):
        """
        Takes a list of tuples, and for each tuple that occurs mutiple times
        marks all but one of the occurences (in the mask that is returned).

        :param tuples: A size (batch, k, rank) tensor of integer tuples
        :return: A size (batch, k) mask indicating the duplicates
        """
        b, k, r = tuples.size()

        primes = self.primes[:r]
        primes = primes.unsqueeze(0).unsqueeze(0).expand(b, k, r)
        unique = ((tuples+1) ** primes).prod(dim=2)  # unique identifier for each tuple

        sorted, sort_idx = torch.sort(unique, dim=1)
        _, unsort_idx = torch.sort(sort_idx, dim=1)

        mask = sorted[:, 1:] == sorted[:, :-1]

        zs = torch.zeros(b, 1, dtype=torch.uint8, device='cuda' if self.use_cuda else 'cpu')
        mask = torch.cat([zs, mask], dim=1)

        return torch.gather(mask, 1, unsort_idx)

    def cuda(self, device_id=None):

        self.use_cuda = True
        super().cuda(device_id)

    def __init__(self, in_num, out_num, k, radditional=0, gadditional=0, region=(128,),
                 sigma_scale=0.2, min_sigma=0.0, fix_value=False):
        super().__init__()

        self.min_sigma = min_sigma
        self.use_cuda = False
        self.in_num = in_num
        self.out_num = out_num
        self.k = k
        self.radditional = radditional
        self.region = region
        self.gadditional = gadditional
        self.sigma_scale = sigma_scale
        self.fix_value = fix_value

        self.weights_rank = 2 # implied rank of W

        self.params = Parameter(torch.randn(k * out_num, 3))

        outs = torch.arange(out_num).unsqueeze(1).expand(out_num, k * (2 + radditional + gadditional)).contiguous().view(-1, 1)
        self.register_buffer('outs', outs.long())

        outs_inf = torch.arange(out_num).unsqueeze(1).expand(out_num, k).contiguous().view(-1, 1)
        self.register_buffer('outs_inf', outs_inf.long())

        self.register_buffer('primes', torch.tensor(util.PRIMES))


    def size(self):
        return (self.out_num, self.in_num)

    def generate_integer_tuples(self, means,rng=None, use_cuda=False):

        dv = 'cuda' if use_cuda else 'cpu'

        c, k, rank = means.size()

        assert rank == 1
        # In the following, we cut the first dimension up into chunks of size self.k (for which the row index)
        # is the same. This then functions as a kind of 'batch' dimension, allowing us to use the code from
        # globalsampling without much adaptation

        """
        Sample the 2 nearest points
        """

        floor_mask = torch.tensor([1, 0], device=dv, dtype=torch.uint8)
        fm = floor_mask.unsqueeze(0).unsqueeze(2).expand(c, k, 2, 1)

        neighbor_ints = means.data.unsqueeze(2).expand(c, k, 2, 1).contiguous()

        neighbor_ints[fm] = neighbor_ints[fm].floor()
        neighbor_ints[~fm] = neighbor_ints[~fm].ceil()

        neighbor_ints = neighbor_ints.long()

        """
        Sample uniformly from a small range around the given index tuple
        """
        rr_ints = torch.cuda.FloatTensor(c, k, self.radditional, 1) if use_cuda else torch.FloatTensor(c, k, self.radditional, 1)

        rr_ints.uniform_()
        rr_ints *= (1.0 - gaussian.EPSILON)

        rng = torch.cuda.FloatTensor(rng) if use_cuda else torch.FloatTensor(rng)

        rngxp = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(rr_ints)  # bounds of the tensor
        rrng = torch.cuda.FloatTensor(self.region) if use_cuda else torch.FloatTensor(self.region)  # bounds of the range from which to sample
        rrng = rrng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(rr_ints)

        mns_expand = means.round().unsqueeze(2).expand_as(rr_ints)

        # upper and lower bounds
        lower = mns_expand - rrng * 0.5
        upper = mns_expand + rrng * 0.5

        # check for any ranges that are out of bounds
        idxs = lower < 0.0
        lower[idxs] = 0.0

        idxs = upper > rngxp
        lower[idxs] = rngxp[idxs] - rrng[idxs]

        rr_ints = (rr_ints * rrng + lower).long()

        """
        Sample uniformly from all index tuples
        """
        g_ints = torch.cuda.FloatTensor(c, k, self.gadditional, 1) if use_cuda else torch.FloatTensor(c, k, self.gadditional, 1)
        rngxp = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(g_ints)  # bounds of the tensor

        g_ints.uniform_()
        g_ints *= (1.0 - gaussian.EPSILON) * rngxp
        g_ints = g_ints.long()

        ints = torch.cat([neighbor_ints, rr_ints, g_ints], dim=2)

        return ints.view(c, -1, rank)

    def forward(self, input, train=True):

        ### Compute and unpack output of hypernetwork

        means, sigmas, values = self.hyper(input)
        nm = means.size(0)
        c = nm // self.k

        means  = means.view(c, self.k, 1)
        sigmas = sigmas.view(c, self.k, 1)
        values = values.view(c, self.k)

        rng = (self.in_num, )

        assert input.size(0) == self.in_num

        if train:
            indices = self.generate_integer_tuples(means, rng=rng, use_cuda=self.use_cuda)
            indfl = indices.float()

            # Mask for duplicate indices
            dups = self.duplicates(indices)

            props = densities(indfl, means, sigmas).clone() # result has size (c, indices.size(1), means.size(1))
            props[dups] = 0
            props = props / props.sum(dim=1, keepdim=True)

            values = values.unsqueeze(1).expand(c, indices.size(1), means.size(1))

            values = props * values
            values = values.sum(dim=2)

            # unroll the batch dimension
            indices = indices.view(-1, 1)
            values = values.view(-1)

            indices = torch.cat([self.outs, indices.long()], dim=1)
        else:
            indices = means.round().long().view(-1, 1)
            values = values.squeeze().view(-1)

            indices = torch.cat([self.outs_inf, indices.long()], dim=1)


        if self.use_cuda:
            indices = indices.cuda()

        # Kill anything on the diagonal
        values[indices[:, 0] == indices[:, 1]] = 0.0

        # if self.symmetric:
        #     # Add reverse direction automatically
        #     flipped_indices = torch.cat([indices[:, 1].unsqueeze(1), indices[:, 0].unsqueeze(1)], dim=1)
        #     indices         = torch.cat([indices, flipped_indices], dim=0)
        #     values          = torch.cat([values, values], dim=0)

        ### Create the sparse weight tensor

        # Prevent segfault
        assert not util.contains_nan(values.data)

        vindices = Variable(indices.t())
        sz = Variable(torch.tensor((self.out_num, self.in_num)))

        spmm = sparsemm(self.use_cuda)
        output = spmm(vindices, values, sz, input)

        return output

    def hyper(self, input=None):
        """
        Evaluates hypernetwork.
        """
        k, width = self.params.size()

        means = F.sigmoid(self.params[:, 0:1])

        # Limits for each of the w_rank indices
        # and scales for the sigmas
        s = torch.cuda.FloatTensor((self.in_num,)) if self.use_cuda else torch.FloatTensor((self.in_num,))
        s = Variable(s.contiguous())

        ss = s.unsqueeze(0)
        sm = s - 1
        sm = sm.unsqueeze(0)

        means = means * sm.expand_as(means)

        sigmas = nn.functional.softplus(self.params[:, 1:2] + gaussian.SIGMA_BOOST) + gaussian.EPSILON

        values = self.params[:, 2:] # * 0.0 + 1.0

        sigmas = sigmas.expand_as(means)
        sigmas = sigmas * ss.expand_as(sigmas)
        sigmas = sigmas * self.sigma_scale + self.min_sigma

        return means, sigmas, values * 0.0 + 1.0/self.k if self.fix_value else values

class GraphConvolution(Module):
    """
    Code adapted from pyGCN, see https://github.com/tkipf/pygcn

    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, has_weight=True):

        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) if has_weight else None
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.weight is not None:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.zero_() # different from the default implementation

    def forward(self, input, adj, train=True):

        if input is None: # The input is the identity matrix
            support = self.weight
        elif self.weight is not None:
            support = torch.mm(input, self.weight)
        else:
            support = input

        output = adj(support, train=train)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ConvModel(nn.Module):

    def __init__(self, data_size, k, radd=32, gadd=32, range=128, min_sigma=0.0):
        super().__init__()

        n, e = data_size

        self.adj = MatrixHyperlayer(n, n, k, radditional=radd, gadditional=gadd, region=(range,),
                            min_sigma=min_sigma, fix_value=True)
    def freeze(self):
        for param in self.encoder_conv.parameters():
            param.requires_grad = False

        for param in self.decoder_conv.parameters():
            param.requires_grad = False

    def forward(self, x, depth=1, train=True):

        n, e = x.size()

        results = []

        for _ in range(1, depth):
            x = self.adj(x, train=train)

            results.append(x)

        return results

    def cuda(self):

        super().cuda()

        self.adj.apply(lambda t: t.cuda())

PLOT_MAX = 2000 # max number of data points for the latent space plot

def go(arg):

    MARGIN = 0.1
    util.makedirs('./conv-simple/')
    torch.manual_seed(arg.seed)

    writer = SummaryWriter()

    data = torch.randn(arg.size, arg.width)

    model = ConvModel(data.size(), k=arg.k,
                      gadd=arg.gadditional, radd=arg.radditional, range=arg.range,
                      min_sigma=arg.min_sigma)

    if arg.cuda:
        model.cuda()
        data = data.cuda()

    data, target = Variable(data), Variable(data)

    optimizer = optim.Adam(list(model.parameters()), lr=arg.lr)
    n, e = data.size()

    for epoch in trange(arg.epochs):

        optimizer.zero_grad()

        outputs = model(data, depth=arg.depth)

        loss = 0.0
        for i, o in enumerate(outputs):
            loss += F.mse_loss(o, data)

        # regularize sigmas
        _, sigmas, _ = model.adj.hyper()

        reg = sigmas.norm().mean()

        # print(loss.item(), reg.item())
        # sys.exit()

        tloss = loss + 0.0001 * reg

        tloss.backward()
        optimizer.step()

        writer.add_scalar('conv-simple/train-tloss', tloss.item(), epoch)
        writer.add_scalar('conv-simple/train-loss', loss.item(), epoch)
        writer.add_scalar('conv-simple/train-reg', reg.item(), epoch)


        if epoch % arg.plot_every == 0:
            print('data')
            print(data[:3, :3].data)
            print()

            for o in outputs:
                print(o[:3, :3].data)

            # Plot the results
            with torch.no_grad():

                outputs = model(data, depth=arg.depth, train=False)

                plt.figure(figsize=(8, 8))

                means, sigmas, values = model.adj.hyper()
                means, sigmas, values = means.data, sigmas.data, values.data
                means = torch.cat([model.adj.outs_inf.data.float(), means], dim=1)

                plt.cla()

                s = model.adj.size()
                util.plot1d(means, sigmas, values.squeeze(), shape=s)
                plt.xlim((-MARGIN * (s[0] - 1), (s[0] - 1) * (1.0 + MARGIN)))
                plt.ylim((-MARGIN * (s[0] - 1), (s[0] - 1) * (1.0 + MARGIN)))

                plt.savefig('./conv-simple/means.{:05}.pdf'.format(epoch))

    print('Finished Training.')

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs",
                        default=1000, type=int)

    parser.add_argument("-W", "--width",
                        dest="width",
                        help="Width of the data.",
                        default=16, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples",
                        default=3, type=int)

    parser.add_argument("-S", "--size",
                        dest="size",
                        help="Number of data points",
                        default=128, type=int)

    parser.add_argument("-a", "--gadditional",
                        dest="gadditional",
                        help="Number of additional points sampled globally per index-tuple",
                        default=32, type=int)

    parser.add_argument("-A", "--radditional",
                        dest="radditional",
                        help="Number of additional points sampled locally per index-tuple",
                        default=16, type=int)

    parser.add_argument("-R", "--range",
                        dest="range",
                        help="Range in which the local points are sampled",
                        default=128, type=int)

    parser.add_argument("-d", "--depth",
                        dest="depth",
                        help="Number of graph convolutions",
                        default=5, type=int)

    parser.add_argument("-p", "--plot-every",
                        dest="plot_every",
                        help="Numer of epochs to wait between plotting",
                        default=100, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.01, type=float)

    parser.add_argument("-r", "--seed",
                        dest="seed",
                        help="Random seed",
                        default=4, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Minimal sigma value",
                        default=0.0, type=float)

    args = parser.parse_args()

    print('OPTIONS', args)

    go(args)
