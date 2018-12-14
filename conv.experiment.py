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
Graph convolution experiment. Given output vectors, learn both the convolution weights and the "graph structure" behind
MNIST.
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

    def __init__(self, data_size, k, emb_size = 16, radd=32, gadd=32, range=128, min_sigma=0.0, directed=True, fix_value=False, encoder=False):
        super().__init__()

        self.data_shape = data_size
        n, c, h, w = data_size

        # - channel sizes
        c1, c2, c3 = 16, 32, 64
        h1, h2, h3 = 256, 128, 64

        # upmode = 'bilinear'
        # self.decoder_conv = nn.Sequential(
        #     nn.Linear(h3, 4 * 4 * c3), nn.ReLU(),
        #     util.Reshape((c3, 4, 4)),
        #     nn.ConvTranspose2d(c3, c3, (3, 3), padding=1), nn.ReLU(),
        #     nn.ConvTranspose2d(c3, c3, (3, 3), padding=1), nn.ReLU(),
        #     nn.ConvTranspose2d(c3, c2, (3, 3), padding=1), nn.ReLU(),
        #     nn.Upsample(scale_factor=3, mode=upmode),
        #     nn.ConvTranspose2d(c2, c2, (3, 3), padding=1), nn.ReLU(),
        #     nn.ConvTranspose2d(c2, c2, (3, 3), padding=1), nn.ReLU(),
        #     nn.ConvTranspose2d(c2, c1, (3, 3), padding=1), nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode=upmode),
        #     nn.ConvTranspose2d(c1, c1, (5, 5), padding=0), nn.ReLU(),
        #     nn.ConvTranspose2d(c1, c1, (3, 3), padding=1), nn.ReLU(),
        #     nn.ConvTranspose2d(c1, 1,  (3, 3), padding=1), nn.Sigmoid(),
        #     # util.Debug(lambda x : print(x.size()))
        # )
        #
        # self.decoder_lin = nn.Sequential(
        #     nn.Linear(emb_size, h3), nn.ReLU(),
        #     nn.Linear(h3, h2), nn.ReLU(),
        #     nn.Linear(h2, h3),
        # )
        #
        # self.decoder = nn.Sequential(
        #     self.decoder_lin,
        #     self.decoder_conv
        # )

        self.decoder = nn.Sequential(
            nn.Linear(emb_size, h3), nn.ReLU(),
            nn.Linear(h3, h2),       nn.ReLU(),
            nn.Linear(h2, h3),       nn.ReLU(),
            nn.Linear(h3, 28*28),    nn.Sigmoid(),
            util.Reshape((1, 28, 28))
        )

        self.encoder = None
        if encoder:
            self.encoder_conv = nn.Sequential(
                nn.Conv2d(1, c1, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c1, c1, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c1, c1, (3, 3), padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(c1, c2, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c2, c2, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c2, c2, (3, 3), padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(c2, c3, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c3, c3, (3, 3), padding=1), nn.ReLU(),
                nn.Conv2d(c3, c3, (3, 3), padding=1), nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                util.Flatten(),
                nn.Linear(9 * c3, h1)
            )

            self.encoder_lin = nn.Sequential(
                util.Flatten(),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, h3), nn.ReLU(),
                nn.Linear(h3, emb_size * 2),
            )

            self.encoder = nn.Sequential(
                self.encoder_conv,
                self.encoder_lin
            )


        self.adj = MatrixHyperlayer(n, n, k, radditional=radd, gadditional=gadd, region=(range,),
                            min_sigma=min_sigma, fix_value=fix_value)

        if not encoder:
            self.embedding = Parameter(torch.randn(n, emb_size))

        self.emb_size = emb_size

        # self.embedding_conv = GraphConvolution(n, emb_size, bias=False)
        # self.weightless_conv = GraphConvolution(emb_size, emb_size, has_weight=False, bias=False)

    def freeze(self):
        for param in self.encoder_conv.parameters():
            param.requires_grad = False

        for param in self.decoder_conv.parameters():
            param.requires_grad = False

    def forward(self, depth=1, train=True, data=None): #, reg=util.kl_batch

        # x0 = self.embedding_conv.weight
        # x = self.embedding_conv(input=None, adj=self.adj, train=train) # identity matrix input
        # results = [x0, x]
        # for i in range(1, depth):
        #     x = self.weightless_conv(input=x, adj=self.adj, train=train)
        #     results.append(x)

        reg_losses = []
        if self.encoder is None:
            x = self.embedding
        else:
            xraw = self.encoder(data)
            xmean, xsig = xraw[:, :self.emb_size], xraw[:, self.emb_size:]
            reg_losses.append(util.kl_loss(xmean, xsig)[:, None])
            x = util.vae_sample(xmean, xsig)

        n, e = x.size()

        results = [x]
        # reg_losses.append(reg(x))

        for _ in range(1, depth):
            x = self.adj(x, train=train)

            results.append(x)
            # reg_losses.append(reg(x))

        # if self.encoder is None:
        #     return [self.decoder(r) for r in results]
        # else:

        return [self.decoder(r) for r in results], None #, reg_losses

    def cuda(self):

        super().cuda()

        self.adj.apply(lambda t: t.cuda())
#
class ConvModelFlat(nn.Module):

    def __init__(self, data_size, k, radd=128, gadd=128, range=16, min_sigma=0.0, fix_value=False):
        super().__init__()

        n, c, h, w = data_size

        self.adj = MatrixHyperlayer(n, n, k, radditional=radd, gadditional=gadd, min_sigma=min_sigma,
                            region=(range,), fix_value=fix_value)


    def forward(self, data, depth=1, train=True):
        n = data.size(0)

        x = data.view(n, -1)
        results =[]
        for _ in range(depth):
            x = self.adj(x, train=train)
            results.append(x)

        return [r.view(data.size()).clamp(0, 1) for r in results]

    def cuda(self):

        super().cuda()

        self.adj.apply(lambda t: t.cuda())

PLOT_MAX = 2000 # max number of data points for the latent space plot

def go(arg):

    MARGIN = 0.1
    util.makedirs('./conv/')
    torch.manual_seed(arg.seed)

    writer = SummaryWriter()

    mnist = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=transforms.ToTensor())

    data = util.totensor(mnist, shuffle=True)

    assert data.min() == 0 and data.max() == 1.0

    if arg.limit is not None:
        data = data[:arg.limit]

    model = ConvModel(data.size(), k=arg.k, emb_size=arg.emb_size,
                      gadd=arg.gadditional, radd=arg.radditional, range=arg.range,
                      min_sigma=arg.min_sigma, fix_value=arg.fix_value, encoder=arg.encoder)

    # model = ConvModelFlat(data.size(), k=arg.k,
    #                   gadd=arg.gadditional, radd=arg.radditional, range=arg.range,
    #                   min_sigma=arg.min_sigma, fix_value=arg.fix_value)

    if arg.cuda:
        model.cuda()
        data = data.cuda()

    data, target = Variable(data), Variable(data)

    optimizer = optim.Adam(
        list(model.parameters()), lr=arg.lr)
    n, c, h, w = data.size()

    for epoch in trange(arg.epochs):

        optimizer.zero_grad()

        depth = 1 if epoch < arg.pretrain_epochs else arg.depth
        if epoch >= arg.pretrain_epochs:
            model.embedding.requires_grad = False

        outputs, _ = model(depth=depth, data=data)

        # reg=lambda x: util.kl_batch(x)[None, None].expand(n, 1) * 10000
        # reg = lambda x: x.norm(dim=1, keepdim=True)
        # reg=lambda x: F.relu(x.norm(dim=1, keepdim=True) - 1.0) * 1000
        # reg=lambda x : torch.zeros(x.size(0), 1)


        rec_losses = []
        for i, o in enumerate(outputs):
            rec_losses.append( F.binary_cross_entropy(o, target, reduce=False).view(n, -1).sum(dim=1, keepdim=True) )
        #
        # losses = torch.cat(rec_losses + reg_losses, dim=1).mean(dim=0)

        # regularize sigmas
        _, sigmas, _ = model.adj.hyper()

        reg = sigmas.norm().mean()

        loss = torch.cat(rec_losses, dim=1).sum()

        # print(loss, reg)
        # sys.exit()

        tloss = loss + 10.0 * reg

        tloss.backward()
        optimizer.step()

        writer.add_scalar('conv/train-loss', loss.item(), epoch)

        if epoch % arg.plot_every == 0:
            plt.figure(figsize=(8, 2))

            # print(losses)
            # if arg.depth > 1:
            #     print('      adj', model.adj.params.grad.mean().item())
            # #  print('    lin', next(model.decoder.parameters()).grad.mean().item())

            plt.cla()
            plt.imshow(np.transpose(torchvision.utils.make_grid(data.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                       interpolation='nearest')
            plt.savefig('./conv/inp.{:05d}.png'.format(epoch))

            # Plot the results
            with torch.no_grad():

                outputs, _ = model(depth=arg.depth, train=False, data=data)

                for d, o in enumerate(outputs):
                    plt.cla()
                    plt.imshow(np.transpose(torchvision.utils.make_grid(o.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                               interpolation='nearest')
                    plt.savefig('./conv/rec.{:05d}.{:02d}.png'.format(epoch, d))

                plt.figure(figsize=(8, 8))

                means, sigmas, values = model.adj.hyper()
                means, sigmas, values = means.data, sigmas.data, values.data
                means = torch.cat([model.adj.outs_inf.data.float(), means], dim=1)

                if arg.draw_matrix:
                    plt.cla()

                    s = model.adj.size()
                    util.plot1d(means, sigmas, values.squeeze(), shape=s)
                    plt.xlim((-MARGIN * (s[0] - 1), (s[0] - 1) * (1.0 + MARGIN)))
                    plt.ylim((-MARGIN * (s[0] - 1), (s[0] - 1) * (1.0 + MARGIN)))

                    plt.savefig('./conv/means.{:05}.pdf'.format(epoch))

                graph = np.concatenate([means.round().long().cpu().numpy(), values.cpu().numpy()], axis=1)
                np.savetxt('graph.{:05}.csv'.format(epoch), graph)

                """
                Plot the data, reconstructions and components
                """
                w, h = 24, 1 + arg.depth + arg.k
                mround = means.round().long()

                plt.figure(figsize=(w, h))

                norm = mpl.colors.Normalize(vmin=-1.0,
                                            vmax=1.0)  # doing this manually, the nx code produces very strange results
                map = mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdYlBu)

                for i in range(w):

                    # plot the image
                    ax = plt.subplot(h, w, i + 1)
                    im = np.transpose(data[i, :, :, :].cpu().numpy(), (1, 2, 0))
                    im = np.squeeze(im)

                    ax.imshow(im, interpolation='nearest', origin='upper', cmap='gray_r')

                    if i == 0:
                        ax.set_ylabel('image')

                    clean(ax)

                    # plot the reconstructions
                    for r, output in enumerate(outputs):
                        ax = plt.subplot(h, w, w*(r+1) +(i + 1))
                        im = np.transpose(output[i, :, :, :].cpu().numpy(), (1, 2, 0))
                        im = np.squeeze(im)

                        ax.imshow(im, interpolation='nearest', origin='upper', cmap='gray_r')

                        if i == 0:
                            ax.set_ylabel('rec. {}'.format(r))

                        clean(ax)

                    # plot the components
                    for c in range(arg.k):
                        ax = plt.subplot(h, w, w*(c+1+len(outputs)) +(i + 1))

                        comp = mround.view(-1, arg.k, 2)[i, c, 1]
                        mult = values.view(-1, arg.k)[i, c]

                        color = np.asarray(map.to_rgba(mult))[:3]

                        im = np.transpose(data[comp, :, :, :].cpu().numpy(), (1, 2, 0))

                        im = im * (1.0 - color)
                        im = 1.0 - im

                        ax.imshow(im,
                                  interpolation='nearest',
                                  origin='upper')

                        clean(ax)

                        if i == 0:
                            ax.set_ylabel('comp. {}'.format(c+1))

                plt.subplots_adjust(wspace=None, hspace=None)
                #fig.tight_layout()
                plt.savefig('./conv/examples{:03}.pdf'.format(epoch), dpi=72)

                """
                Plot the embeddings
                """

                if arg.depth == 2:
                    map = None
                else:
                    norm = mpl.colors.Normalize(vmin=1.0, vmax=arg.depth)
                    map = mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.tab10)

                latents = model.encoder(data) if arg.encoder else model.embedding.data

                images = data.data.cpu().permute(0, 2, 3, 1).numpy()[:PLOT_MAX, :]

                ax = None
                size = None
                for d in range(arg.depth):

                    if d == 0:
                        color = None
                    elif map is None:
                        color = np.asarray([0.0, 0.0, 1.0, 1.0])
                    else:
                        color = map.to_rgba(d)

                    l2 = latents[:PLOT_MAX, :2]

                    ax, size = util.scatter_imgs(l2.cpu().numpy(), images, ax=ax, color=color, size=size)

                    if d < arg.depth - 1:
                        latents = model.adj(latents, train=False)

                util.clean(ax)
                plt.savefig('./conv/latent-space.{:05}.pdf'.format(epoch), dpi=600)

                """
                Plot the graph (reasonable results for small datasets)
                """
                if arg.draw_graph:
                    # Plot the graph
                    outputs = model(depth=arg.depth, train=False)

                    g = nx.MultiDiGraph()
                    g.add_nodes_from(range(data.size(0)))

                    print('Drawing graph at ', epoch, 'epochs')
                    for i in range(means.size(0)):
                        m = means[i, :].round().long()
                        v = values[i]

                        g.add_edge(m[1].item(), m[0].item(), weight=v.item() )
                        # print(m[1].item(), m[0].item(), v.item())

                    plt.figure(figsize=(8,8))
                    ax = plt.subplot(111)

                    pos = nx.spring_layout(g, iterations=100, k=5/math.sqrt(data.size(0)))
                    # pos = nx.circular_layout(g)

                    nx.draw_networkx_nodes(g, pos, node_size=30, node_color='w', node_shape='s', axes=ax)
                    # edges = nx.draw_networkx_edges(g, pos, edge_color=values.data.view(-1), edge_vmin=0.0, edge_vmax=1.0, cmap='bone')

                    weights = [d['weight'] for (_, _, d) in g.edges(data=True)]

                    colors = map.to_rgba(weights)

                    nx.draw_networkx_edges(g, pos, width=1.0, edge_color=colors, axes=ax)

                    ims = 0.03
                    xmin, xmax = float('inf'), float('-inf')
                    ymin, ymax = float('inf'), float('-inf')

                    out0 = outputs[1].data
                    # out1 = outputs[1].data

                    for i, coords in pos.items():

                        extent  = (coords[0] - ims, coords[0] + ims, coords[1] - ims, coords[1] + ims)
                        # extent0 = (coords[0] - ims, coords[0] + ims, coords[1] + ims, coords[1] + 3 * ims)
                        # extent1 = (coords[0] - ims, coords[0] + ims, coords[1] + 3 * ims, coords[1] + 5 * ims)

                        ax.imshow(data[i].cpu().squeeze(), cmap='gray_r', extent=extent,  zorder=100, alpha=1)
                        # ax.imshow(out0[i].cpu().squeeze(),  cmap='pink_r', extent=extent0, zorder=100, alpha=0.85)
                        # ax.imshow(out1[i].cpu().squeeze(),  cmap='pink_r', extent=extent1, zorder=100)

                        xmin, xmax = min(coords[0], xmin), max(coords[0], xmax)
                        ymin, ymax = min(coords[1], ymin), max(coords[1], ymax)

                    MARGIN = 0.3
                    ax.set_xlim(xmin-MARGIN, xmax+MARGIN)
                    ax.set_ylim(ymin-MARGIN, ymax+MARGIN)

                    plt.axis('off')

                    plt.savefig('./conv/graph.{:05}.pdf'.format(epoch), dpi=300)

    print('Finished Training.')

def test():
    """
    Poor man's unit test
    """

    indices = Variable(torch.tensor([[0,1],[1,0],[2,1]]), requires_grad=True)
    values = Variable(torch.tensor([1.0, 2.0, 3.0]), requires_grad=True)
    size = Variable(torch.tensor([3, 2]))

    wsparse = torch.sparse.FloatTensor(indices.t(), values, (3,2))
    wdense  = Variable(torch.tensor([[0.0,1.0],[2.0,0.0],[0.0, 3.0]]), requires_grad=True)
    x = Variable(torch.randn(2, 4), requires_grad=True)
    #
    # print(wsparse)
    # print(wdense)
    # print(x)

    # dense version
    mul = torch.mm(wdense, x)
    loss = mul.norm()
    loss.backward()

    print('dw', wdense.grad)
    print('dx', x.grad)

    del loss

    # spmm version
    # mul = torch.mm(wsparse, x)
    # loss = mul.norm()
    # loss.backward()
    #
    # print('dw', values.grad)
    # print('dx', x.grad)

    x.grad = None
    values.grad = None

    mul = util.SparseMultCPU.apply(indices.t(), values, size, x)
    loss = mul.norm()
    loss.backward()

    print('dw', values.grad)
    print('dx', x.grad)

    # Finite elements approach for w
    for h in [1e-4, 1e-5, 1e-6]:
        grad = torch.zeros(values.size(0))
        for i in range(values.size(0)):
            nvalues = values.clone()
            nvalues[i] = nvalues[i] + h

            mul = util.SparseMultCPU.apply(indices.t(), values, size, x)
            loss0 = mul.norm()

            mul = util.SparseMultCPU.apply(indices.t(), nvalues, size, x)
            loss1 = mul.norm()

            grad[i] = (loss1-loss0)/h

        print('hw', h, grad)

    # Finite elements approach for x
    for h in [1e-4, 1e-5, 1e-6]:
        grad = torch.zeros(x.size())
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                nx = x.clone()
                nx[i, j] = x[i, j] + h

                mul = util.SparseMultCPU.apply(indices.t(), values, size, x)
                loss0 = mul.norm()

                mul = util.SparseMultCPU.apply(indices.t(), values, size, nx)
                loss1 = mul.norm()

                grad[i, j] = (loss1-loss0)/h

        print('hx', h, grad)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--test", dest="test",
                        help="Run the unit tests.",
                        action="store_true")

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs",
                        default=250, type=int)

    parser.add_argument("-P", "--pretrain-epochs",
                        dest="pretrain_epochs",
                        help="Number of epochs spent optimizing the embeddings. After this cutoff, the embeddings are frozen, and the adjacency matrix is trained.",
                        default=1000, type=int)

    parser.add_argument("-E", "--emb_size",
                        dest="emb_size",
                        help="Size of the node embeddings.",
                        default=16, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples",
                        default=3, type=int)

    parser.add_argument("-L", "--limit",
                        dest="limit",
                        help="Number of data points",
                        default=None, type=int)

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
    #
    # parser.add_argument("-S", "--undirected", dest="undirected",
    #                     help="Use an undirected graph",
    #                     action="store_true")

    parser.add_argument("-J", "--draw-matrix", dest="draw_matrix",
                        help="Draw the adjacency matrix",
                        action="store_true")

    parser.add_argument("-G", "--draw-graph", dest="draw_graph",
                        help="Draw the graph",
                        action="store_true")

    parser.add_argument("-F", "--fix-value", dest="fix_value",
                        help="Fix the values of the matrix to 1",
                        action="store_true")

    parser.add_argument("-N", "--use-encoder", dest="encoder",
                        help="Whether to use an encoder",
                        action="store_true")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data')

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Minimal sigma value",
                        default=0.0, type=float)

    parser.add_argument("-Q", "--regularization-weight",
                        dest="regweight",
                        help="Regularization weight (the bigger this is, the faster the sigma's converge).",
                        default=1.0, type=float)

    args = parser.parse_args()

    if args.test:
        test()
        print('Tests completed succesfully.')
        sys.exit()

    print('OPTIONS', args)

    go(args)
