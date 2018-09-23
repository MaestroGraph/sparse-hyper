import gaussian
import util, time, itertools
from gaussian import Bias

import torch, random
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from util import Lambda, Debug

import torch.optim as optim
import sys

import torchvision
import torchvision.transforms as transforms

from util import od

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser

import networkx as nx

"""
Graph convolution experiment. Given output vectors, learn both the convolution weights and the "graph structure" behind
MNIST.
"""

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
#
def sparsemult(use_cuda):
    return SparseMultGPU.apply if use_cuda else SparseMultCPU.apply

class SparseMultCPU(torch.autograd.Function):

    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, xmatrix):

        # print(type(size), size, list(size), intlist(size))
        # print(indices.size(), values.size(), torch.Size(intlist(size)))

        matrix = torch.sparse.FloatTensor(indices, values, torch.Size(util.intlist(size)))

        ctx.indices, ctx.matrix, ctx.xmatrix = indices, matrix, xmatrix

        return torch.mm(matrix, xmatrix)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data

        # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices

        i_ixs = ctx.indices[0,:]
        j_ixs = ctx.indices[1,:]
        output_select = grad_output[i_ixs, :]
        xmatrix_select = ctx.xmatrix[j_ixs, :]

        grad_values = (output_select * xmatrix_select).sum(dim=1)

        grad_xmatrix = torch.mm(ctx.matrix.t(), grad_output)
        return None, Variable(grad_values), None, Variable(grad_xmatrix)

class SparseMultGPU(torch.autograd.Function):

    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, xmatrix):

        # print(type(size), size, list(size), intlist(size))

        matrix = torch.cuda.sparse.FloatTensor(indices, values, torch.Size(util.intlist(size)))

        ctx.indices, ctx.matrix, ctx.xmatrix = indices, matrix, xmatrix

        return torch.mm(matrix, xmatrix)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data

        # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices

        i_ixs = ctx.indices[0,:]
        j_ixs = ctx.indices[1,:]
        output_select = grad_output[i_ixs]
        xmatrix_select = ctx.xmatrix[j_ixs]

        grad_values = (output_select *  xmatrix_select).sum(dim=1)

        grad_xmatrix = torch.mm(ctx.matrix.t(), grad_output)
        return None, Variable(grad_values), None, Variable(grad_xmatrix)

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

    n, d, rank = points.size()

    means = means.unsqueeze(1).expand_as(points)

    sigmas = sigmas.unsqueeze(1).expand_as(points)
    sigmas_squared = torch.sqrt(1.0/(gaussian.EPSILON+sigmas))

    points = points - means
    points = points * sigmas_squared

    # Compute dot products for all points
    points = points.view(-1, 1, rank)
    # -- dot prod
    products = torch.bmm(points, points.transpose(1,2))
    # -- reconstruct shape
    products = products.view(n, d)

    num = torch.exp(- 0.5 * products)

    return num

class MatrixHyperlayer(nn.Module):
    """
    The normal hyperlayer samples a sparse matrix for each input in the batch. In a graph convolution we don't have
    batches, but we do have multiple inputs, so we rewrite the hyperlayer in a highly simplified form.
    """

    def cuda(self, device_id=None):

        self.use_cuda = True
        super().cuda(device_id)

        self.floor_mask = self.floor_mask.cuda()

    def __init__(self, in_num, out_num, k, additional=0, sigma_scale=0.2, min_sigma=0.0, symmetric=True):
        super().__init__()

        self.min_sigma = min_sigma
        self.use_cuda = False
        self.in_num = in_num
        self.out_num = out_num
        self.additional = additional
        self.sigma_scale = sigma_scale
        self.symmetric = symmetric

        self.weights_rank = 2 # implied rank of W

        # create a matrix with all binary sequences of length 'rank' as rows
        lsts = [[int(b) for b in bools] for bools in itertools.product([True, False], repeat=self.weights_rank)]
        self.floor_mask = torch.ByteTensor(lsts)

        self.params   = Parameter(torch.randn(k, 4))

    def size(self):
        return (self.out_num, self.in_num)

    def discretize(self, means, sigmas, values, rng=None, additional=16, use_cuda=False):
        """
        Takes the output of a hypernetwork (real-valued indices and corresponding values) and turns it into a list of
        integer indices, by "distributing" the values to the nearest neighboring integer indices.

        NB: the returned ints is not a Variable (just a plain LongTensor). autograd of the real valued indices passes
        through the values alone, not the integer indices used to instantiate the sparse matrix.

        :param ind: A Variable containing a matrix of N by K, where K is the number of indices.
        :param val: A Variable containing a vector of length N containing the values corresponding to the given indices
        :return: a triple (ints, props, vals). ints is an N*2^K by K matrix representing the N*2^K integer index-tuples that can
            be made by flooring or ceiling the indices in 'ind'. 'props' is a vector of length N*2^K, which indicates how
            much of the original value each integer index-tuple receives (based on the distance to the real-valued
            index-tuple). vals is vector of length N*2^K, containing the value of the corresponding real-valued index-tuple
            (ie. vals just repeats each value in the input 'val' 2^K times).
        """

        n, rank = means.size()

        # ints is the same size as ind, but for every index-tuple in ind, we add an extra axis containing the 2^rank
        # integerized index-tuples we can make from that one real-valued index-tuple
        # ints = torch.cuda.FloatTensor(batchsize, n, 2 ** rank + additional, rank) if use_cuda else FloatTensor(batchsize, n, 2 ** rank, rank)
        t0 = time.time()

        # BATCH_NEIGHBORS approach
        fm = self.floor_mask.unsqueeze(0).expand(n, 2 ** rank, rank)

        neighbor_ints = means.data.unsqueeze(1).expand(n, 2 ** rank, rank).contiguous()

        neighbor_ints[fm] = neighbor_ints[fm].floor()
        neighbor_ints[~fm] = neighbor_ints[~fm].ceil()

        neighbor_ints = neighbor_ints.long()

        # Sample additional points
        if rng is not None:
            t0 = time.time()
            total = util.prod(rng)

            # not gaussian.PROPER_SAMPLING (since it's a big matrix)
            sampled_ints = torch.cuda.FloatTensor(n, additional, rank) if use_cuda else torch.FloatTensor(n, additional, rank)

            sampled_ints.uniform_()
            sampled_ints *= (1.0 - gaussian.EPSILON)

            rng = torch.cuda.FloatTensor(rng) if use_cuda else torch.FloatTensor(rng)
            rng = rng.unsqueeze(0).unsqueeze(0).expand_as(sampled_ints)

            sampled_ints = torch.floor(sampled_ints * rng).long()

            ints = torch.cat((neighbor_ints, sampled_ints), dim=1)

            ints_fl = ints.float()

        ints_fl = Variable(ints_fl)  # leaf node in the comp graph, gradients go through values

        # compute the proportion of the value each integer index tuple receives
        props = densities(ints_fl, means, sigmas)

        # -- normalize the proportions for each index tuple
        sums = torch.sum(props + gaussian.EPSILON, dim=1, keepdim=True).expand_as(props)
        props = props / sums

        # repeat each value 2^rank+A times, so it matches the new indices
        val = values.expand_as(props).contiguous()

        # 'Unroll' the ints tensor into a long list of integer index tuples (ie. a matrix of (n*(2^rank+add)) by rank)
        ints = ints.view(-1, rank)

        # ... and reshape the proportions and values the same way
        props = props.view(-1)
        val = val.view(-1)

        return ints, props, val

    def forward(self, input, train=True):

        ### Compute and unpack output of hypernetwork

        means, sigmas, values = self.hyper(input)

        t0total = time.time()

        rng = (self.out_num, self.in_num)

        assert input.size(0) == self.in_num

        # turn the real values into integers in a differentiable way
        t0 = time.time()

        if train:
            indices, props, values = self.discretize(means, sigmas, values, rng=rng, additional=self.additional, use_cuda=self.use_cuda)

            values = values * props
        else:
            indices = means.round().long()
            values = values.squeeze()

        if self.use_cuda:
            indices = indices.cuda()

        # # diagonal
        # dindices = torch.tensor(range(self.in_num), device='cuda' if self.use_cuda else 'cpu').unsqueeze(1).expand(self.in_num, 2)
        #
        # indices = torch.cat([indices, dindices], dim=0)
        # values = torch.cat([values, self.diagonal], dim=0)

        # Kill anything on the diagonal
        values[indices[:, 0] == indices[:, 1]] = 0.0

        if self.symmetric:
            # Add reverse direction automatically
            flipped_indices = torch.cat([indices[:, 1].unsqueeze(1), indices[:, 0].unsqueeze(1)], dim=1)
            indices         = torch.cat([indices, flipped_indices], dim=0)
            values          = torch.cat([values, values], dim=0)

        ### Create the sparse weight tensor

        t0 = time.time()

        # Prevent segfault
        assert not util.contains_nan(values.data)

        # print(vindices.size(), bfvalues.size(), bfsize, bfx.size())
        vindices = Variable(indices.t())
        sz = Variable(torch.tensor(rng))

        spmm = sparsemult(self.use_cuda)
        output = spmm(vindices, values, sz, input)

        return output

    def hyper(self, input=None):
        """
        Evaluates hypernetwork.
        """
        k, width = self.params.size()
        w_rank = width - 2

        means = F.sigmoid(self.params[:, 0:w_rank])

        ## expand the indices to the range [0, max]

        # Limits for each of the w_rank indices
        # and scales for the sigmas
        ws = (self.out_num, self.in_num)
        s = torch.cuda.FloatTensor(ws) if self.use_cuda else torch.FloatTensor(ws)
        s = Variable(s.contiguous())

        ss = s.unsqueeze(0)
        sm = s - 1
        sm = sm.unsqueeze(0)

        means = means * sm.expand_as(means)

        sigmas = nn.functional.softplus(self.params[:, w_rank:w_rank + 1] + gaussian.SIGMA_BOOST) + gaussian.EPSILON

        values = self.params[:, w_rank + 1:] # * 0.0 + 1.0

        sigmas = sigmas.expand_as(means)
        sigmas = sigmas * ss.expand_as(sigmas)
        sigmas = sigmas * self.sigma_scale + self.min_sigma

        return means, sigmas, F.tanh(values)


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

    def forward(self, input, adj : MatrixHyperlayer, train=True):

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

    def __init__(self, data_size, k, emb_size = 16, additional=128, min_sigma=0.0):
        super().__init__()

        self.data_shape = data_size
        n, c, h, w = data_size

        ch1, ch2, ch3 = 128, 64, 32
        # # decoder from embedding to images
        # self.decoder= nn.Sequential(
        #     nn.Linear(emb_size, 4 * 4 * ch1), nn.ReLU(),
        #     util.Reshape((ch1, 4, 4)),
        #     nn.ConvTranspose2d(ch1, ch1, (5, 5), padding=2), nn.ReLU(),
        #     nn.ConvTranspose2d(ch1, ch1, (5, 5), padding=2), nn.ReLU(),
        #     nn.ConvTranspose2d(ch1, ch2, (5, 5), padding=2), nn.ReLU(),
        #     nn.Upsample(scale_factor=3, mode='bilinear'),
        #     nn.ConvTranspose2d(ch2, ch2, (5, 5), padding=2), nn.ReLU(),
        #     nn.ConvTranspose2d(ch2, ch2, (5, 5), padding=2), nn.ReLU(),
        #     nn.ConvTranspose2d(ch2, ch1, (5, 5), padding=2), nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.ConvTranspose2d(ch1, ch1, (5, 5), padding=2), nn.ReLU(),
        #     nn.ConvTranspose2d(ch1, ch1, (5, 5), padding=2), nn.ReLU(),
        #     nn.ConvTranspose2d(ch1, 1, (5, 5), padding=0), nn.Sigmoid()
        # )

        self.decoder = nn.Sequential(
            nn.Linear(emb_size, 28*28),
            #nn.Linear(256, 28*28),
            nn.Sigmoid(), util.Reshape((1, 28, 28))
        )

        self.adj = MatrixHyperlayer(n,n, k, additional=additional, min_sigma=min_sigma)
        self.embedding = Parameter(torch.randn(n, emb_size))

        # self.embedding_conv = GraphConvolution(n, emb_size, bias=False)
        # self.weightless_conv = GraphConvolution(emb_size, emb_size, has_weight=False, bias=False)

    def forward(self, depth=1, train=True):

        # x0 = self.embedding_conv.weight
        # x = self.embedding_conv(input=None, adj=self.adj, train=train) # identity matrix input
        # results = [x0, x]
        # for i in range(1, depth):
        #     x = self.weightless_conv(input=x, adj=self.adj, train=train)
        #     results.append(x)

        x = self.embedding
        results =[x]
        for _ in range(1, depth):
            x = self.adj(x)
            results.append(x)

        return [self.decoder(r) for r in results]

    def cuda(self):

        super().cuda()

        self.adj.apply(lambda t: t.cuda())

class ConvModelFlat(nn.Module):

    def __init__(self, data_size, k, emb_size = 16, additional=128, min_sigma=0.0, directed=True):
        super().__init__()

        n, c, h, w = data_size

        self.adj = MatrixHyperlayer(n, n, k, additional=additional, min_sigma=min_sigma, symmetric=not directed)

    def forward(self, data, depth=1, train=True):
        n = data.size(0)

        x = data.view(n, -1)
        results =[]
        for _ in range(depth):
            x = self.adj(x, train=train)
            results.append(x)

        return [r.view(data.size()) for r in results]

    def cuda(self):

        super().cuda()

        self.adj.apply(lambda t: t.cuda())

def go(arg):

    MARGIN = 0.1
    util.makedirs('./conv/')
    torch.manual_seed(arg.seed)

    w = SummaryWriter()

    SHAPE = (1, 28, 28)

    mnist = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=transforms.ToTensor())
    data = util.totensor(mnist, shuffle=True)

    clusters = [[0, 3, 4, 14], [1, 7, 10, 11], [8, 15]]
    gold = []

    for c in clusters:
        for ii, i in enumerate(c):
            for j in c[ii+1:]:
                gold.append([i, j])

    gold = Variable(torch.tensor(gold, dtype=torch.float))

    assert data.min() == 0 and data.max() == 1.0

    if arg.limit is not None:
        data = data[:arg.limit]

    model = ConvModelFlat(data.size(), k=arg.k, emb_size=arg.emb_size, additional=arg.additional, min_sigma=arg.min_sigma, directed=not arg.undirected)

    if arg.cuda: # This probably won't work (but maybe with small data)
        model.cuda()
        data = data.cuda()

    data, target = Variable(data), Variable(data)

    ## SIMPLE
    optimizer = optim.Adam(model.parameters(), lr=arg.lr)

    for epoch in trange(arg.epochs):

        optimizer.zero_grad()

        outputs = model(data, depth=arg.depth)

        losses = torch.zeros((len(outputs),), device='cuda' if arg.cuda else 'cpu')
        for i, o in enumerate(outputs):
            losses[i] = (F.mse_loss(o, target))

        loss = losses.sum()

        # means, sigmas, values = model.adj.hyper()
        # cheatloss = F.mse_loss(means, gold)
        #
        # (loss * 0.00001 + cheatloss).backward()  # compute the gradients

        loss.backward()
        optimizer.step()

        w.add_scalar('mnist/train-loss', loss.item(), epoch)

        if epoch % arg.plot_every == 0:

            print('{:03}   '.format(epoch), losses)
            print('    adj', model.adj.params.grad.mean().item())
            #  print('    lin', next(model.decoder.parameters()).grad.mean().item())

            plt.cla()
            plt.imshow(np.transpose(torchvision.utils.make_grid(data.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                       interpolation='nearest')
            plt.savefig('./conv/inp.{:03d}.png'.format(epoch))

            outputs = model(data, depth=arg.depth)

            for d, o in enumerate(outputs):

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(o.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('./conv/rec.{:03d}.{:02d}.png'.format(epoch, d))

            plt.figure(figsize=(7, 7))

            means, sigmas, values = model.adj.hyper()

            plt.cla()

            s = model.adj.size()
            util.plot(means.unsqueeze(0), sigmas.unsqueeze(0), values.unsqueeze(0).squeeze(2), shape=s)
            plt.xlim((-MARGIN * (s[0] - 1), (s[0] - 1) * (1.0 + MARGIN)))
            plt.ylim((-MARGIN * (s[0] - 1), (s[0] - 1) * (1.0 + MARGIN)))

            plt.savefig('./conv/means{:03}.pdf'.format(epoch))

            # Plot the graph
            outputs = model(data, depth=arg.depth, train=False)

            g = nx.MultiGraph() if arg.undirected else nx.MultiDiGraph()
            g.add_nodes_from(range(data.size(0)))

            means, _, values = model.adj.hyper()

            print('Drawing graph at ', epoch, 'epochs')
            elist = []
            for i in range(means.size(0)):
                m = means[i, :].round().long()
                v = values[i]

                g.add_edge(m[1].item(), m[0].item(), weight=v.item() )
                # print(m[1].item(), m[0].item(), v.item())

            plt.figure(figsize=(8,8))
            ax = plt.subplot(111)

            pos = nx.spring_layout(g, iterations=500)
            # pos = nx.circular_layout(g)

            nx.draw_networkx_nodes(g, pos, node_size=30, node_color='w', node_shape='s', axes=ax)
            # edges = nx.draw_networkx_edges(g, pos, edge_color=values.data.view(-1), edge_vmin=0.0, edge_vmax=1.0, cmap='bone')

            weights = [d['weight'] for (_, _, d) in g.edges(data=True)]

            norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0) # doing this manually, the nx code produces very strange results
            map = mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdYlBu)

            colors = map.to_rgba(weights)

            nx.draw_networkx_edges(g, pos, width=1.0, edge_color=colors, axes=ax)

            ims = 0.03
            xmin, xmax = float('inf'), float('-inf')
            ymin, ymax = float('inf'), float('-inf')

            out0 = outputs[0].data
            # out1 = outputs[1].data

            for i, coords in pos.items():

                extent  = (coords[0] - ims, coords[0] + ims, coords[1] - ims, coords[1] + ims)
                extent0 = (coords[0] - ims, coords[0] + ims, coords[1] + ims, coords[1] + 3 * ims)
                # extent1 = (coords[0] - ims, coords[0] + ims, coords[1] + 3 * ims, coords[1] + 5 * ims)

                ax.imshow(data[i].cpu().squeeze(), cmap='gray_r', extent=extent,  zorder=100)
                ax.imshow(out0[i].cpu().squeeze(),  cmap='pink_r', extent=extent0, zorder=100)
                # ax.imshow(out1[i].cpu().squeeze(),  cmap='pink_r', extent=extent1, zorder=100)

                xmin, xmax = min(coords[0], xmin), max(coords[0], xmax)
                ymin, ymax = min(coords[1], ymin), max(coords[1], ymax)

            MARGIN = 0.1
            ax.set_xlim(xmin-MARGIN, xmax+MARGIN)
            ax.set_ylim(ymin-MARGIN, ymax+MARGIN)

            plt.axis('off')

            plt.savefig('./conv/graph{:03}.pdf'.format(epoch), dpi=300)

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

    mul = SparseMultCPU.apply(indices.t(), values, size, x)
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

            mul = SparseMultCPU.apply(indices.t(), values, size, x)
            loss0 = mul.norm()

            mul = SparseMultCPU.apply(indices.t(), nvalues, size, x)
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

                mul = SparseMultCPU.apply(indices.t(), values, size, x)
                loss0 = mul.norm()

                mul = SparseMultCPU.apply(indices.t(), values, size, nx)
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

    parser.add_argument("-E", "--emb_size",
                        dest="emb_size",
                        help="Size of the node embeddings.",
                        default=16, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples",
                        default=80000, type=int)

    parser.add_argument("-L", "--limit",
                        dest="limit",
                        help="Number of data points",
                        default=None, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled oper index-tuple",
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

    parser.add_argument("-S", "--undirected", dest="undirected",
                        help="Use an undirected graph",
                        action="store_true")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data')

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Minimal sigma value",
                        default=0.0, type=float)

    args = parser.parse_args()

    if args.test:
        test()
        print('Tests completed succesfully.')
        sys.exit()

    print('OPTIONS', args)

    go(args)
