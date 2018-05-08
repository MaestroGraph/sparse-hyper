import hyper, gaussian, util, time, pretrain, os, math, sys
import torch, random
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import Parameter
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from util import Lambda, Debug

from torch.utils.data import TensorDataset, DataLoader

from torchsample.metrics import CategoricalAccuracy

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from util import od, prod

from argparse import ArgumentParser

import networkx as nx

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

"""
Graph experiment

"""

class GraphEncoder(gaussian.HyperLayer):
    """
    A graph-specific ASH layer. Applies a single hypernetwork to each edge in the graph,
    generating k index tuples for that specific edge.

    """

    def __init__(self, nodes, out_shape, k, additional=0, sigma_scale=0.1, num_values=-1, min_sigma=0.0, subsample=None):

        super().__init__(in_rank=2, out_shape=out_shape, additional=additional, bias_type=gaussian.Bias.DENSE, subsample=subsample)

        self.n = nodes

        self.k = k # the number of index tuples _per edge in the input_
        self.sigma_scale = sigma_scale
        self.num_values = num_values
        self.out_shape= out_shape
        self.min_sigma = min_sigma

        outsize = k * (2 + len(out_shape) + 2)

        activation = nn.ReLU()

        hidden = nodes

        self.source = nn.Sequential(
            nn.Linear(nodes * 2, hidden), # graph edges in 1-hot encoding
            # activation,
            # nn.Linear(hidden, hidden),
            # activation,
            # nn.Linear(hidden, hidden),
            # activation,
            # nn.Linear(hidden, hidden),
            # activation,
            # nn.Linear(hidden, hidden),
            # activation,
            # nn.Linear(hidden, hidden),
            activation,
            nn.Linear(hidden, outsize),
        )

        self.bias = Parameter(torch.zeros(*out_shape))

        if num_values > 0:
            self.values = Parameter(torch.randn((num_values,)))

    def forward(self, input):
        dense, sparse = input

        means, sigmas, values, bias = self.hyper(sparse)

        return self.forward_inner(dense, means, sigmas, values, bias)

    def hyper(self, sparse):
        """
        Evaluates hypernetwork.
        """

        b, num_edges, n2 = sparse.size()
        sparse = sparse.view(b*num_edges, n2)

        res = self.source(sparse).unsqueeze(2).view(b, self.k * num_edges, 2 + len(self.out_shape) + 2)

        means, sigmas, values = self.split_out(res, (self.n, self.n), self.out_shape)

        sigmas = sigmas * self.sigma_scale + self.min_sigma

        if self.num_values > 0:
            mult = self.k // self.num_values

            values = self.values.unsqueeze(0).expand(mult, self.num_values)
            values = values.contiguous().view(-1)[:self.k]

            values = values.unsqueeze(0).unsqueeze(0).expand(b, num_edges, self.k)
            values = values.contiguous().view(b, self.k*num_edges)

        self.last_values = values.data

        return means, sigmas, values, self.bias

class GraphDecoder(gaussian.HyperLayer):
    """
    Takes a latent vector as input and decodes to a graph output.

    For each of the k index tuples, we take the input, concatenate it with a length k one hot encoding of the tuple-index
    and pass it through an MLP.


    """

    def __init__(self, input_size, out_shape, k, additional=0, sigma_scale=0.1, num_values=-1, min_sigma=0.0, subsample=None):

        super().__init__(in_rank=1, out_shape=out_shape, additional=additional, bias_type=gaussian.Bias.DENSE, subsample=subsample)

        self.n = input_size

        self.k = k
        self.sigma_scale = sigma_scale
        self.num_values = num_values
        self.out_shape = out_shape
        self.min_sigma = min_sigma

        activation = nn.ReLU()

        hidden = input_size

        self.source = nn.Sequential(
            nn.Linear(input_size + k, hidden),
            activation,
            nn.Linear(hidden, hidden),
            activation,
            # nn.Linear(hidden, hidden),
            # activation,
            nn.Linear(hidden, 1  + len(out_shape) + 2),
        )

        self.bias = Parameter(torch.zeros(*out_shape))

        if num_values > 0:
            self.values = Parameter(torch.randn((num_values,)))

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        b, n = input.size()

        input = input.unsqueeze(1).expand(b, self.k, n)

        id = torch.eye(self.k).unsqueeze(0).expand(b, self.k, self.k)
        if input.is_cuda:
            id = id.cuda()
        id = Variable(id)

        input = torch.cat([input, id], dim=2)
        input = input.view(b * self.k, n + self.k)

        res = self.source(input).unsqueeze(1).view(b, self.k, 1 + len(self.out_shape) + 2)

        means, sigmas, values = self.split_out(res, (self.n,), self.out_shape)
        sigmas = sigmas * self.sigma_scale + self.min_sigma

        if self.num_values > 0:
            mult = self.k // self.num_values
            values = self.values.unsqueeze(0).expand(mult, self.num_values)
            values = values.contiguous().view(-1)[:self.k]

            values = values.unsqueeze(0).expand(b, self.k)

        self.last_values = values.data

        return means, sigmas, values, self.bias

def vae_loss(x, x_rec, mu, logvar, variational=True):
    b, w, h = x.size()
    total = util.prod(x.size()[1:])

    xent = nn.functional.binary_cross_entropy(x_rec.contiguous().view(-1, total), x.contiguous().view(-1, total))

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / (b * total)

    return xent + kl

def generate_er(n=128, m=512, num=64):

    dense = torch.FloatTensor(num, n, n)
    sparse = torch.zeros(num, m, n*2)

    for i in range(num):
        graph = nx.gnm_random_graph(n, m)
        am = nx.to_numpy_matrix(graph)

        dense[i, :, :] = torch.from_numpy(am)

        for j, (fr, to) in enumerate(graph.edges):
            sparse[i, j, fr] = 1.0
            sparse[i, j, n + to] = 1.0

    return dense, sparse

SIZE = 600
PLOT = True

def go(nodes=128, links=512, batch=64, epochs=350, k=750, kpe=7, additional=512, modelname='baseline', cuda=False,
       seed=1, bias=True, lr=0.001, lambd=0.01, subsample=None, num_values=-1, min_sigma=0.0, adaptive_decoder=False,
       tb_dir=None, variational_epoch=0, zsize=32):

    FT = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    torch.manual_seed(seed)

    w = SummaryWriter(log_dir=tb_dir)

    SHAPE = (1, nodes, nodes)

    LOG.info('generating data...')
    dense, sparse = generate_er(nodes, links, SIZE)
    LOG.info('done.')

    if modelname == 'basic':

        encoder = GraphEncoder(nodes, (zsize * 2,), k=kpe, additional=additional, subsample=subsample, num_values=num_values, min_sigma=min_sigma)

        if adaptive_decoder:
            decoder = GraphDecoder(zsize, SHAPE, k=k, additional=additional,
                                    subsample=subsample, num_values=num_values, min_sigma=min_sigma)
        else:
            decoder = gaussian.ParamASHLayer((zsize, ), SHAPE, k=k, additional=additional, has_bias=bias,
                                    subsample=subsample,  min_sigma=min_sigma)
        # decoder = nn.Linear(zsize, util.prod(SHAPE))

        if cuda:
            encoder.cuda()
            decoder.cuda()
    else:
        raise Exception('Model name {} not recognized'.format(modelname))

    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=lr)

    step = 0
    iterations = int(math.ceil(SIZE/batch))

    variational = False

    sigs_enc, sigs_dec = [], []
    vals_enc, vals_dec = [], []


    for epoch in range(epochs):

        if epoch == variational_epoch:
            variational = True

        for i in trange(iterations):

            # get the inputs
            f, t = i * batch, (i+1)*batch if i < iterations - 1 else SIZE
            batch_dense = dense[f:t, :, :]
            batch_sparse = sparse[f:t, :, :]

            if cuda:
                batch_dense = batch_dense.cuda()
                batch_sparse = batch_sparse.cuda()

            batch_dense  = Variable(batch_dense)
            batch_sparse = Variable(batch_sparse)

            optimizer.zero_grad()

            h = encoder((batch_dense, batch_sparse))

            mu, logvar = h[:, zsize:], h[:, zsize:]
            sample = Variable(FT(mu.size()).normal_())

            std = logvar.mul(0.5).exp()
            sample = sample.mul(std).add(mu)

            sample = sample

            reconstruction = decoder(sample if variational else mu.contiguous())
            reconstruction = nn.functional.sigmoid(reconstruction)

            loss = vae_loss(batch_dense, reconstruction, mu, logvar)

            t0 = time.time()
            loss.backward()  # compute the gradients
            logging.info('backward: {} seconds'.format(time.time() - t0))

            optimizer.step()

            w.add_scalar('graphs/train-loss', loss.data[0], step)

            step += t - f

            if PLOT and i == 0:

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(batch_dense.unsqueeze(1).data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('g1vae.{:03d}.input.pdf'.format(epoch))

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(reconstruction.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('g1vae.{:03d}.output.pdf'.format(epoch))

                sigmas = list(encoder.last_sigmas[0, :])
                values = list(encoder.last_values[0, :])

                sigs_enc.append(sigmas)
                vals_enc.append(values)

                ax = plt.figure().add_subplot(111)

                for j, (s, v) in enumerate(zip(sigs_enc, vals_enc)):
                    ax.scatter([j] * len(s), s, c=v, linewidth=0,  alpha=0.2, cmap='RdYlBu', vmin=-1.0, vmax=1.0)

                ax.set_aspect('auto')
                plt.ylim(ymin=0)
                util.clean()

                plt.savefig('sigmas.enc.pdf')
                plt.savefig('sigmas.enc.png')

                sigmas = list(decoder.last_sigmas[0, :])
                values = list(decoder.last_values[0, :])

                sigs_dec.append(sigmas)
                vals_dec.append(values)

                ax = plt.figure().add_subplot(111)
                for j, (s, v) in enumerate(zip(sigs_dec, vals_dec)):

                    ax.scatter([j] * len(s), s, c=v, linewidth=0, alpha=0.2, cmap='RdYlBu', vmin=-1.0, vmax=1.0)
                plt.ylim(ymin=0)


                util.clean()

                plt.savefig('sigmas.dec.pdf')
                plt.savefig('sigmas.dec.png')

    LOG.info('Finished Training.')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-N", "--nodes",
                        dest="nodes",
                        help="Number of nodes in the generated graphs.",
                        default=128, type=int)

    parser.add_argument("-M", "--links",
                        dest="links",
                        help="Number of links in the generated graphs.",
                        default=256, type=int)

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        help="Number of epochs over thegenerated data.",
                        default=350, type=int)

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Which model to train.",
                        default='basic')

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-K", "--num-points-per-edge",
                        dest="kpe",
                        help="Number of index tuples per edge in the graph layer ",
                        default=7, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples in the decoder layer",
                        default=750, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled",
                        default=512, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-B", "--no-bias", dest="bias",
                        help="Whether to give the layers biases.",
                        action="store_false")

    parser.add_argument("-V", "--variational", dest="variational",
                        help="The epoch at which to start the variational training",
                        default=0, type=int)

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data')

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-L", "--lambda",
                        dest="lambd",
                        help="Reconstruction loss weight",
                        default=0.01, type=float)

    parser.add_argument("-S", "--subsample",
                        dest="subsample",
                        help="Sample a subset of the indices to estimate gradients for",
                        default=None, type=float)

    parser.add_argument("-F", "--num-values", dest="num_values",
                        help="How many fixed values to allow the network",
                        default=-1, type=int)

    parser.add_argument("-A", "--adaptive-decoder",
                        dest="adaptive_decoder",
                        help="Whether to use an adaptive hypernetwork for the decoder.",
                        action="store_true")

    parser.add_argument("-W", "--min-sigma",
                        dest="min_sigma",
                        help="Minimum value of sigma.",
                        default=0.0, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)

    parser.add_argument("-Z", "--zsize",
                        dest="zsize",
                        help="Size of the latent vector.",
                        default=16, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(epochs=options.epochs, batch=options.batch_size, nodes=options.nodes, links=options.links, k=options.k, kpe=options.kpe, bias=options.bias,
        additional=options.additional, modelname=options.model, cuda=options.cuda,
        lr=options.lr, lambd=options.lambd, subsample=options.subsample,
        num_values=options.num_values, min_sigma=options.min_sigma, adaptive_decoder=options.adaptive_decoder,
        tb_dir=options.tb_dir, variational_epoch=options.variational, zsize=options.zsize)
