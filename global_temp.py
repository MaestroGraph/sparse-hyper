import torch
from numpy.core.multiarray import dtype
from torch.autograd import Variable
from torch.nn import Parameter
#from torch import FloatTensor, LongTensor

import abc, itertools, math, types
from numpy import prod

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from util import *
import util

import sys
import time, random, logging

from enum import Enum

from tqdm import trange

from gaussian import Bias, fi_matrix, flatten_indices_mat, tup, fi

# added to the sigmas to prevent NaN
EPSILON = 10e-7
PROPER_SAMPLING = False # NB: set to true for very small tranformations.
BATCH_NEIGHBORS = True # Faster way of finding neighbor index-tuples
SIGMA_BOOST = 2.0

"""
Version of the hyperlayer that learns only over a subset of the columns of the matrix. The rest of the matrix is hardwired.

This can, for instance, be used to create a layer that has 3 incoming connections for each node in the output layer, with
the connections to the input nodes beaing learned.

This version comnbines samples for all index tuples that have the same fixed indices. For instance, is we have fixed
 columns and we are learning across rows, we sample and distribute globally within each row.

This requires that the template have equal-sized, contiguous chunks of index tuples for which the fixed values are the
   same.
"""


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

    b, k, l, rank = points.size()
    b, k, c, rank = means.size()

    #--
    points = points.unsqueeze(3).expand(b, k, l, c, rank)
    means  = means.unsqueeze(2).expand_as(points)
    sigmas = sigmas.unsqueeze(2).expand_as(points)

    sigmas_squared = torch.sqrt(1.0/(EPSILON+sigmas))

    points = points - means
    points = points * sigmas_squared

    # print(points.size())
    # sys.exit()

    # Compute dot products for all points
    # -- unroll the batch/k/l/c dimensions
    points = points.view(-1, 1, rank)
    # -- dot prod
    products = torch.bmm(points, points.transpose(1, 2))
    # -- reconstruct shape
    products = products.view(b, k, l, c)

    num = torch.exp(- 0.5 * products) # the numerator of the Gaussian density

    return num

class HyperLayer(nn.Module):
    """
        Abstract class for the hyperlayer. Implement by defining a hypernetwork, and returning it from the hyper() method.
    """
    @abc.abstractmethod
    def hyper(self, input):
        """
            Applies the hypernetwork. This network should take the same input as the hyperlayer itself
            and output a pair (L, V), with L a matrix of k by R (with R the rank of W) and a vector V of length k.
        """
        return

    def cuda(self, device_id=None):

        self.use_cuda = True
        super().cuda(device_id)

        self.floor_mask = self.floor_mask.cuda()

    def __init__(self, in_rank, out_size, temp_indices, learn_cols, chunk_size, gadditional=0, radditional=0, region=None,
                 bias_type=Bias.DENSE, sparse_input=False, subsample=None):
        """

        :param in_rank:
        :param temp_indices: This should be a tensor (not a variable). It contains all hardwired connections
        :param lean_cols: A tuple of integers indicating which columns in the index matrix must be learned
        :param additional:
        :param bias_type:
        :param sparse_input:
        :param subsample:
        """
        super().__init__()

        self.use_cuda = False
        self.in_rank = in_rank
        self.out_size = out_size # without batch dimension
        self.gadditional = gadditional
        self.radditional = radditional
        self.region = region

        self.bias_type = bias_type
        self.sparse_input = sparse_input
        self.subsample = subsample
        self.learn_cols = learn_cols
        self.chunk_size = chunk_size

        # create a tensor with all binary sequences of length 'out_rank' as rows
        # (this will be used to compute the nearby integer-indices of a float-index).
        lsts = [[int(b) for b in bools] for bools in itertools.product([True, False], repeat=len(learn_cols))]
        self.floor_mask = torch.ByteTensor(lsts)

        # template for the index matrix containing the hardwired connections
        # The learned parts can be set to zero; they will be overriden.
        assert temp_indices.size(1) == in_rank + len(out_size)

        self.register_buffer('temp_indices', temp_indices)

        self.register_buffer('primes', torch.tensor(
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]))

    def duplicates(self, tuples):
        """
        Takes a list of tuples, and for each tuple that occurs mutiple times
        marks all but one of the occurences (in the mask that is returned), across dim 2.

        :param tuples: A size (batch, k, rank) tensor of integer tuples
        :return: A size (batch, k) mask indicating the duplicates
        """
        b, k, l, r = tuples.size()

        primes = self.primes[:r]
        primes = primes[None, None, None, :].expand(b, k, l, r)
        unique = ((tuples+1) ** primes).prod(dim=3)  # unique identifier for each tuple

        sorted, sort_idx = torch.sort(unique, dim=2)
        _, unsort_idx = torch.sort(sort_idx, dim=2) # get the idx required to reverse the sort

        mask = sorted[:, :, 1:] == sorted[:, :, :-1]

        zs = torch.zeros(b, k, 1, dtype=torch.uint8, device='cuda' if self.use_cuda else 'cpu')
        mask = torch.cat([zs, mask], dim=2)

        return torch.gather(mask, 2, unsort_idx)

    def split_out(self, res, size):
        """
        Utility function. res is a B x K x Wrank+2 tensor with range from
        -inf to inf, this function splits out the means, sigmas and values, and
        applies the required activations.

        :param res:
        :param size:
        :param output_size:
        :param gain:
        :return:
        """

        b, k, width = res.size()
        w_rank = width - 2

        assert w_rank == len(self.learn_cols)

        means = nn.functional.sigmoid(res[:, :, 0:w_rank])
        means = means.unsqueeze(2).contiguous().view(-1, k, w_rank)

        ## expand the indices to the range [0, max]

        # Limits for each of the w_rank indices
        # and scales for the sigmas
        ws = list(size)
        s = torch.cuda.FloatTensor(ws) if self.use_cuda else torch.FloatTensor(ws)
        s = Variable(s.contiguous())

        ss = s.unsqueeze(0).unsqueeze(0)
        sm = s - 1
        sm = sm.unsqueeze(0).unsqueeze(0)

        means = means * sm.expand_as(means)

        sigmas = nn.functional.softplus(res[:, :, w_rank:w_rank + 1] + SIGMA_BOOST).squeeze(2) + EPSILON

        values = res[:, :, w_rank + 1:].squeeze(2)

        self.last_sigmas = sigmas.data
        self.last_values = values.data

        sigmas = sigmas.unsqueeze(2).expand_as(means)
        sigmas = sigmas * ss.expand_as(sigmas)

        return means, sigmas, values

    def generate_integer_tuples(self, means, rng=None, use_cuda=False, relative_range=None):
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

        b, k, c, rank = means.size()

        # ints is the same size as ind, but for every index-tuple in ind, we add an extra axis containing the 2^rank
        # integerized index-tuples we can make from that one real-valued index-tuple
        # ints = torch.cuda.FloatTensor(batchsize, n, 2 ** rank + additional, rank) if use_cuda else FloatTensor(batchsize, n, 2 ** rank, rank)

        """
        Generate nearby tuples
        """
        fm = self.floor_mask[None, None, None, :].expand(b, k, c, 2 ** rank, rank)

        neighbor_ints = means.data[:, :, :, None, :].expand(b, k, c, 2 ** rank, rank).contiguous()

        neighbor_ints[fm] = neighbor_ints[fm].floor()
        neighbor_ints[~fm] = neighbor_ints[~fm].ceil()

        neighbor_ints = neighbor_ints.long()

        """
        Sample uniformly from all integer tuples
        """

        sampled_ints = torch.cuda.FloatTensor(b, k, c, self.gadditional, rank) if use_cuda \
            else torch.FloatTensor(b, k, c, self.gadditional, rank)

        sampled_ints.uniform_()
        sampled_ints *= (1.0 - EPSILON)

        rng = torch.cuda.FloatTensor(rng) if use_cuda else torch.FloatTensor(rng)
        rngxp = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(sampled_ints)

        sampled_ints = torch.floor(sampled_ints * rngxp).long()

        """
        Sample uniformly from a small range around the given index tuple
        """
        rr_ints = torch.cuda.FloatTensor(b, k, c, self.radditional, rank) if use_cuda \
            else torch.FloatTensor(b, k, c,  self.radditional, rank)

        rr_ints.uniform_()
        rr_ints *= (1.0 - EPSILON)

        rngxp = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(rr_ints) # bounds of the tensor
        rrng = torch.cuda.FloatTensor(relative_range) if use_cuda \
            else torch.FloatTensor(relative_range) # bounds of the range from which to sample

        rrng = rrng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(rr_ints)

        # print(means.size())
        mns_expand = means.round().unsqueeze(3).expand_as(rr_ints)

        # upper and lower bounds
        lower = mns_expand - rrng * 0.5
        upper = mns_expand + rrng * 0.5

        # check for any ranges that are out of bounds
        idxs = lower < 0.0
        lower[idxs] = 0.0

        idxs = upper > rngxp
        lower[idxs] = rngxp[idxs] - rrng[idxs]

        # print('means', means.round().long())
        # print('lower', lower)

        rr_ints = (rr_ints * rrng + lower).long()

        all = torch.cat([neighbor_ints, sampled_ints, rr_ints] , dim=3)

        return all.view(b, k, -1, rank) # combine all indices sampled within a chunk

    def forward(self, input, **kwargs):

        ### Compute and unpack output of hypernetwork

        t0 = time.time()
        bias = None

        if self.bias_type == Bias.NONE:
            means, sigmas, values = self.hyper(input, **kwargs)
        elif self.bias_type == Bias.DENSE:
            means, sigmas, values, bias = self.hyper(input, **kwargs)
        elif self.bias_type == Bias.SPARSE:
            means, sigmas, values, bias_means, bias_sigmas, bias_values = self.hyper(input, **kwargs)
        else:
            raise Exception('bias type {} not recognized.'.format(self.bias_type))

        logging.info('compute hyper: {} seconds'.format(time.time() - t0))

        if self.sparse_input:
            input = input.dense()

        return self.forward_inner(input, means, sigmas, values, bias)

    def forward_inner(self, input, means, sigmas, values, bias):

        b, n, r = means.size()

        k = n//self.chunk_size
        c = self.chunk_size
        means, sigmas, values = means.view(b, k, c, r), sigmas.view(b, k, c, r), values.view(b, k, c)

        batchsize = input.size()[0]

        # turn the real values into integers in a differentiable way
        # max values allowed for each colum in the index matrix
        fullrange = self.out_size + input.size()[1:]
        subrange = [fullrange[r] for r in self.learn_cols]

        indices = self.generate_integer_tuples(means, rng=subrange, use_cuda=self.use_cuda, relative_range=self.region)
        indfl = indices.float()

        # Mask for duplicate indices
        dups = self.duplicates(indices)

        props = densities(indfl, means, sigmas).clone()  # result has size (b, k, l, c), l = indices[2]
        props[dups, :] = 0
        props = props / props.sum(dim=2, keepdim=True)

        values = values[:, :, None, :].expand_as(props)

        values = props * values
        values = values.sum(dim=3)

        indices, values = indices.view(b, -1 , r), values.view(b, -1)

        # stitch it into the template
        b, l, r = indices.size()
        h, w = self.temp_indices.size()
        template = self.temp_indices[None, :, None, :].expand(b, h, l//h, w)
        template = template.contiguous().view(b, l, w)

        template[:, :, self.learn_cols] = indices
        indices = template

        if self.use_cuda:
            indices = indices.cuda()

        # translate tensor indices to matrix indices

        # mindices, flat_size = flatten_indices(indices, input.size()[1:], self.out_shape, self.use_cuda)
        mindices, flat_size = flatten_indices_mat(indices, input.size()[1:], self.out_size)

        # NB: mindices is not an autograd Variable. The error-signal for the indices passes to the hypernetwork
        #     through 'values', which are a function of both the real_indices and the real_values.

        ### Create the sparse weight tensor

        x_flat = input.view(batchsize, -1)

        sparsemult = util.sparsemult(self.use_cuda)

        # Prevent segfault
        try:
            assert mindices.min() >= 0
            assert not util.contains_nan(values.data)
        except AssertionError as ae:
            print('Nan in values or negative index in mindices.')
            print('means', means)
            print('sigmas', sigmas)
            print('props', props)
            print('values', values)
            print('indices', indices)
            print('mindices', mindices)

            raise ae

        # Then we flatten the batch dimension as well
        bm = util.bmult(flat_size[1], flat_size[0], mindices.size()[1], batchsize, self.use_cuda)
        bfsize = Variable(flat_size * batchsize)

        bfindices = mindices + bm
        bfindices = bfindices.view(1, -1, 2).squeeze(0)
        vindices = Variable(bfindices.t())

        #- bfindices is now a sparse representation of a big block-diagonal matrix (nxb times mxb), with the batches along the
        #  diagonal (and the rest zero). We flatten x over all batches, and multiply by this to get a flattened y.

        # print(bfindices.size(), flat_size)
        # print(bfindices)

        bfvalues = values.view(1, -1).squeeze(0)
        bfx = x_flat.view(1, -1).squeeze(0)

        bfy = sparsemult(vindices, bfvalues, bfsize, bfx)

        y_flat = bfy.unsqueeze(0).view(batchsize, -1)

        y_shape = [batchsize]
        y_shape.extend(self.out_size)

        y = y_flat.view(y_shape) # reshape y into a tensor

        ### Handle the bias
        if self.bias_type == Bias.DENSE:
            y = y + bias
        if self.bias_type == Bias.SPARSE:
            raise Exception('Not implemented yet.')

        return y
