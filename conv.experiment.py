import gaussian
import util, logging, time, itertools
from gaussian import Bias

import torch, random
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from util import Lambda, Debug

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from util import od

from argparse import ArgumentParser

"""
Graph convolution experiment. Given output vectors, learn both the convolution weights and the "graph structure" behind
MNIST.
"""

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class MatrixHyperlayer(nn.Module):
    """
    The normal hyperlayer samples a sparse matrix for each input in the batch. In a graph convolution we don't have
    batches, but we do have multiple inputs, so we rewrite the hyperlayer in a highly simplified form.
    """

    def cuda(self, device_id=None):

        self.use_cuda = True
        super().cuda(device_id)

        self.floor_mask = self.floor_mask.cuda()

    def __init__(self, in_num, out_num, k, additional=0, sigma_scale=0.2):
        super().__init__()

        self.use_cuda = False
        self.in_num = in_num
        self.out_num = out_num
        self.additional = additional
        self.sigma_scale = sigma_scale

        self.weights_rank = 2 # implied rank of W

        # create a matrix with all binary sequences of length 'rank' as rows
        lsts = [[int(b) for b in bools] for bools in itertools.product([True, False], repeat=self.weights_rank)]
        self.floor_mask = torch.ByteTensor(lsts)

        self.params = Parameter(torch.randn(k, 4))

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

        logging.info('  neighbors: {} seconds'.format(time.time() - t0))

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

            logging.info('  sampling: {} seconds'.format(time.time() - t0))

        ints_fl = Variable(ints_fl)  # leaf node in the comp graph, gradients go through values

        t0 = time.time()
        # compute the proportion of the value each integer index tuple receives
        props = gaussian.densities(ints_fl, means, sigmas)
        # props is batchsize x K x 2^rank+a, giving a weight to each neighboring or sampled integer-index-tuple

        # -- normalize the proportions of the neigh points and the
        sums = torch.sum(props + gaussian.EPSILON, dim=2, keepdim=True).expand_as(props)
        props = props / sums

        logging.info('  densities: {} seconds'.format(time.time() - t0))
        t0 = time.time()

        # repeat each value 2^rank+A times, so it matches the new indices
        val = torch.unsqueeze(values, 2).expand_as(props).contiguous()

        # 'Unroll' the ints tensor into a long list of integer index tuples (ie. a matrix of n*2^rank by rank for each
        # instance in the batch) ...
        ints = ints.view(-1, rank, 1).squeeze(3)

        # ... and reshape the proportions and values the same way
        # props = props.view(batchsize, -1)
        # val = val.view(batchsize, -1)

        logging.info('  reshaping: {} seconds'.format(time.time() - t0))

        return ints, props, val

    def forward(self, input):

        ### Compute and unpack output of hypernetwork

        t0 = time.time()

        means, sigmas, values = self.hyper(input)

        logging.info('compute hyper: {} seconds'.format(time.time() - t0))

        t0total = time.time()

        rng = (self.out_num, self.in_num)

        assert input.size(0) == self.in_num

        # turn the real values into integers in a differentiable way
        t0 = time.time()

        indices, props, values = self.discretize(means, sigmas, values, rng=rng, additional=self.additional, use_cuda=self.use_cuda)

        values = values * props

        logging.info('discretize: {} seconds'.format(time.time() - t0))

        if self.use_cuda:
            indices = indices.cuda()

        # translate tensor indices to matrix indices
        t0 = time.time()

        logging.info('flatten: {} seconds'.format(time.time() - t0))

        # NB: mindices is not an autograd Variable. The error-signal for the indices passes to the hypernetwork
        #     through 'values', which are a function of both the real_indices and the real_values.

        ### Create the sparse weight tensor

        sparsemult = util.sparsemult(self.use_cuda)

        t0 = time.time()

        # Prevent segfault
        assert not util.contains_nan(values.data)

        # print(vindices.size(), bfvalues.size(), bfsize, bfx.size())
        output = sparsemult(indices, values, rng, input)

        logging.info('sparse mult: {} seconds'.format(time.time() - t0))

        logging.info('total: {} seconds'.format(time.time() - t0total))

        return output

    def hyper(self, input):
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

        values = self.params[:, w_rank + 1:]

        sigmas = sigmas.expand_as(means)
        sigmas = sigmas * ss.expand_as(sigmas)
        sigmas = sigmas * self.sigma_scale # + self.min_sigma

        return means, sigmas, values


class GraphConvolution(Module):
    """
    Code adapted from pyGCN, see https://github.com/tkipf/pygcn

    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj : MatrixHyperlayer):

        if input is None: # The input is the identity matrix
            support = self.weight
        else:
            support = torch.mm(input, self.weight)

        output = adj(support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ConvModel(nn.Module):

    def __init__(self, data_size, k, emb_size = 16, depth=2, additional=128):
        super().__init__()

        self.data_shape = data_size
        n, c, h, w = data_size

        ch1, ch2, ch3 = 128, 64, 32
        # decoder from embedding to images
        self.decoder= nn.Sequential(
            nn.Linear(emb_size, 4 * 4 * ch1), nn.ReLU(),
            util.Reshape((ch1, 4, 4)),
            nn.ConvTranspose2d(ch1, ch1, (5, 5), padding=2), nn.ReLU(),
            nn.ConvTranspose2d(ch1, ch1, (5, 5), padding=2), nn.ReLU(),
            nn.ConvTranspose2d(ch1, ch2, (5, 5), padding=2), nn.ReLU(),
            nn.Upsample(scale_factor=3, mode='bilinear'),
            nn.ConvTranspose2d(ch2, ch2, (5, 5), padding=2), nn.ReLU(),
            nn.ConvTranspose2d(ch2, ch2, (5, 5), padding=2), nn.ReLU(),
            nn.ConvTranspose2d(ch2, ch1, (5, 5), padding=2), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(ch1, ch1, (5, 5), padding=2), nn.ReLU(),
            nn.ConvTranspose2d(ch1, ch1, (5, 5), padding=2), nn.ReLU(),
            nn.ConvTranspose2d(ch1, 1, (5, 5), padding=0), nn.Sigmoid()
        )

        self.adj = MatrixHyperlayer(n,n, k, additional=additional)

        self.convs = [GraphConvolution(n, emb_size)]
        for _ in range(1, depth):
            self.convs.append(GraphConvolution(emb_size, emb_size))

    def forward(self):

        x = self.convs[0](input=None, adj=self.adj) # identity matrix input
        for _ in range(1, len(self.convs)):
            x = F.sigmoid(x)
            x = self.conv2(input=x, adj=self.adj)

        return self.decoder(x)

    def cuda(self):

        super().cuda()

        for hyper in self.hyperlayers:
            hyper.apply(lambda t: t.cuda())

def go(arg):

    torch.manual_seed(arg.seed)
    logging.basicConfig(filename='run.log',level=logging.INFO)

    w = SummaryWriter()

    SHAPE = (1, 28, 28)

    mnist = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=transforms.ToTensor())
    data = util.totensor(mnist, shuffle=True)

    assert data.min() == 0 and data.max() == 1.0

    if arg.limit is not None:
        data = data[:arg.limit]

    model = ConvModel(data.size(), k=arg.k, emb_size=arg.emb_size, depth=arg.depth, additional=arg.additional)

    if arg.cuda: # This probably won't work (but maybe with small data)
        model.cuda()
        data = data.cuda()

    data = Variable(data)

    ## SIMPLE
    optimizer = optim.Adam(model.parameters(), lr=arg.lr)

    for epoch in trange(arg.epochs):


        optimizer.zero_grad()

        outputs = model()
        loss = F.binary_cross_entropy(outputs, data)

        t0 = time.time()
        loss.backward()  # compute the gradients
        logging.info('backward: {} seconds'.format(time.time() - t0))
        optimizer.step()

        w.add_scalar('mnist/train-loss', loss.item(), epoch)

        print(epoch, loss.item())

    print('Finished Training.')

if __name__ == "__main__":

    parser = ArgumentParser()

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

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data')

    args = parser.parse_args()

    print('OPTIONS', args)

    go(args)
