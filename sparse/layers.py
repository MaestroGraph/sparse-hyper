import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch import FloatTensor, LongTensor

import abc, itertools, math, types
from numpy import prod

import torch.nn.functional as F

import tensors

from sparse.util import Bias, sparsemult, contains_nan, bmult, nduplicates

import sys
import random

from enum import Enum

# added to the sigmas to prevent NaN
EPSILON = 10e-7
SIGMA_BOOST = 2.0


"""
Core implementation of the sparse (hyper)layer as an abstract class (SparseLayer).

"""

def densities(points, means, sigmas):
    """
    Compute the unnormalized probability densities of a given set of points for a
    given set of multivariate normal distrbutions (MVNs)

    :param means: (b, n, c, r) tensor of n vectors of dimension r (in a batch of size b)
        representing the means of n MVNs
    :param sigmas: (b, k, l, r) tensor of n vectors of dimension r (in a batch of size b)
        representing the diagonal covariance matrix of n MVNs
    :param points: The points for which to compute the probabilioty densities
    :return: (b, k, n) tensor containing the density of every point under every MVN
    """

    # n: number of MVNs
    # d: number of points per MVN
    # rank: dim of points

    b, k, l, rank = points.size()
    b, k, c, rank = means.size()

    points = points.unsqueeze(3).expand(b, k, l, c, rank)
    means  = means.unsqueeze(2).expand_as(points)
    sigmas = sigmas.unsqueeze(2).expand_as(points)

    sigmas_squared = torch.sqrt(1.0/(EPSILON+sigmas))

    points = points - means
    points = points * sigmas_squared

    # Compute dot products for all points
    # -- unroll the batch/k/l/c dimensions
    points = points.view(-1, 1, rank)
    # -- dot prod
    products = torch.bmm(points, points.transpose(1, 2))
    # -- reconstruct shape
    products = products.view(b, k, l, c)

    num = torch.exp(- 0.5 * products) # the numerator of the Gaussian density

    return num

def transform_means(means, size):
    """
    Transforms raw parameters for the index tuples (with values in (-inf, inf)) into parameters within the bound of the
    dimensions of the tensor.

    In the case of a templated sparse layer, these parameters and the corresponing size tuple deascribe only the learned
    subtensor.

    :param means: (batch, k, rank) tensor of raw parameter values
    :param size: Tuple describing the tensor dimensions.
    :return:
    """
    b, k, rank = means.size()

    # Scale to [0, 1]
    means = F.sigmoid(means)

    # Compute upper bounds
    s = torch.tensor(list(size), dtype=torch.float, device='cuda' if means.is_cuda else 'cpu') - 1
    s = s[None, None, :].expand_as(means)

    return means * s

def transform_sigmas(sigmas, size, min_sigma=EPSILON):
    """
    Transforms raw parameters for the conv matrices (with values in (-inf, inf)) into positive values, scaled proportional
    to the dimensions of the tensor. Note: each sigma is parametrized by a single value, which is expanded to a vector to
    fit the diagonal of the covariance matrix.

    In the case of a templated sparse layer, these parameters and the corresponing size tuple deascribe only the learned
    subtensor.

    :param sigmas: (batch, k) matrix of raw sigma values
    :param size: Tuple describing the tensor dimensions.
    :param min_sigma: Minimal sigma value.
    :return:
    """
    b, k = sigmas.size()
    r = len(size)

    # Scale to [0, 1]
    sigmas = F.softplus(sigmas + SIGMA_BOOST) + min_sigma
    sigmas = sigmas[:, :, None].expand(b, k, r)

    # Compute upper bounds
    s = torch.tensor(list(size), dtype=torch.float, device='cuda' if sigmas.is_cuda else 'cpu')
    s = s[None, None, :].expand_as(sigmas)

    return sigmas * s

class SparseLayer(nn.Module):
    """
    Abstract class for the templated hyperlayer. Implement by defining a hypernetwork, and returning it from the
    hyper() method. See NASLayer for an implementation without hypernetwork.

    The templated hyperlayer takes certain columns of its index-tuple matrix as fixed (the template), and others as
    learnable. Imagine a neural network layer where the connections to the output nodes are fixed, but the connections to
    the input nodes can be learned. For a non-templated hypernetwork, just leave the template parameters None.
    """
    @abc.abstractmethod
    def hyper(self, input):
        """
        Applies the hypernetwork, and returns the continuous index tuples, with their associated sigmas and values.

        :param input: The input to the hyperlayer.
        :return: A triple: (means, sigmas, values)
        """
        raise NotImplementedError

    def __init__(self, in_rank, out_size,
                 temp_indices=None,
                 learn_cols=None,
                 chunk_size=None,
                 gadditional=0, radditional=0, rrange=None,
                 bias_type=Bias.DENSE):
        """
        :param in_rank: Nr of dimensions in the input. The specific size may vary between inputs.
        :param out_size: Tuple describing the size of the output.
        :param temp_indices: The template describing the fixed part of the tuple index-tuple matrix. None for a
            non-templated hyperlayer.
        :param learn_cols: Which columns of the template are 'free' (to be learned). The rest are fixed. None for a
            non-templated hyperlayer.
        :param chunk_size: Size of the "context" of generating integer index tuples. Duplicates are removed withing the
            same context. The list of continuous index tuples is chunked into contexts of this size. If none, the whole
            list counts as a single context. This is mostly useful in combination with templating.
        :param gadditional: Number of points to sample globally per index tuple
        :param radditional: Number of points to sample locally per index tuple
        :param rrange: Tuple describing the size of the region over which the local additional points are sampled (must
            be smaller than the size of the tensor).
        :param bias_type: The type of bias of the sparse layer (none, dense or sparse).
        :param subsample:
        """

        super().__init__()
        rank = in_rank + len(out_size)

        self.use_cuda = False
        self.in_rank = in_rank
        self.out_size = out_size # without batch dimension
        self.gadditional = gadditional
        self.radditional = radditional
        self.region = rrange
        self.chunk_size = chunk_size

        self.bias_type = bias_type
        self.learn_cols = learn_cols if learn_cols is not None else range(rank)

        self.templated = temp_indices is not None

        # create a tensor with all binary sequences of length 'out_rank' as rows
        # (this will be used to compute the nearby integer-indices of a float-index).
        lsts = [[int(b) for b in bools] for bools in itertools.product([True, False], repeat=len(self.learn_cols))]
        self.register_buffer('floor_mask', torch.ByteTensor(lsts))

        if self.templated:
            # template for the index matrix containing the hardwired connections
            # The learned parts can be set to zero; they will be overriden.
            assert temp_indices.size(1) == in_rank + len(out_size)

            self.register_buffer('temp_indices', temp_indices)

    def generate_integer_tuples(self, means, rng=None, use_cuda=False, relative_range=None):
        """
        Take continuous-valued index tuples, and generates integer-valued index tuples.

        The returned matrix of ints is not a Variable (just a plain LongTensor). autograd of the real valued indices passes
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
        dv = 'cuda' if use_cuda else 'cpu'
        FT = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        # ints is the same size as ind, but for every index-tuple in ind, we add an extra axis containing the 2^rank
        # integerized index-tuples we can make from that one real-valued index-tuple
        # ints = torch.cuda.FloatTensor(batchsize, n, 2 ** rank + additional, rank) if use_cuda else FloatTensor(batchsize, n, 2 ** rank, rank)

        """
        Generate neighbor tuples
        """
        fm = self.floor_mask[None, None, None, :].expand(b, k, c, 2 ** rank, rank)

        neighbor_ints = means.data[:, :, :, None, :].expand(b, k, c, 2 ** rank, rank).contiguous()

        neighbor_ints[fm] = neighbor_ints[fm].floor()
        neighbor_ints[~fm] = neighbor_ints[~fm].ceil()

        neighbor_ints = neighbor_ints.long()

        """
        Sample uniformly from all integer tuples
        """

        sampled_ints = FT(b, k, c, self.gadditional, rank)

        sampled_ints.uniform_()
        sampled_ints *= (1.0 - EPSILON)

        rng = FT(rng)
        rngxp = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(sampled_ints)

        sampled_ints = torch.floor(sampled_ints * rngxp).long()

        """
        Sample uniformly from a small range around the given index tuple
        """
        rr_ints = FT(b, k, c, self.radditional, rank)

        rr_ints.uniform_()
        rr_ints *= (1.0 - EPSILON)

        rngxp = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(rr_ints) # bounds of the tensor
        rrng = FT(relative_range) # bounds of the range from which to sample

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

    def forward(self, input, mrange=None, seed=None, **kwargs):

        ### Compute and unpack output of hypernetwork

        bias = None

        if self.bias_type == Bias.NONE:
            means, sigmas, values = self.hyper(input, **kwargs)
        elif self.bias_type == Bias.DENSE:
            means, sigmas, values, bias = self.hyper(input, **kwargs)
        elif self.bias_type == Bias.SPARSE:
            raise Exception('Sparse bias not supported yet.')
        else:
            raise Exception('bias type {} not recognized.'.format(self.bias_type))

        b, n, r = means.size()

        c = self.chunk_size if self.chunk_size is not None else n
        k = n // self.chunk_size if self.chunk_size is not None else 1

        means, sigmas, values = means.view(b, k, c, r), sigmas.view(b, k, c, r), values.view(b, k, c)

        assert b == input.size(0), 'input batch size ({}) should match parameter batch size ({}).'.format(input.size(0), b)

        # max values allowed for each column in the index matrix
        fullrange = self.out_size + input.size()[1:]
        subrange = [fullrange[r] for r in self.learn_cols]  # submatrix for the learnable dimensions

        if not self.training:
            indices = means.squeeze().round().long()
        else:
            indices = self.generate_integer_tuples(means, rng=subrange, use_cuda=self.use_cuda, relative_range=self.region)
            indfl = indices.float()

            # Mask for duplicate indices
            dups = nduplicates(indices)

            # compute (unnormalized) densities under the given MVNs (proportions)
            props = densities(indfl, means, sigmas).clone()  # result has size (b, k, l, c), l = indices[2]
            props[dups, :] = 0
            props = props / props.sum(dim=2, keepdim=True) # normalize over all points of a given index tuple

            # Weight the values by the proportions
            values = values[:, :, None, :].expand_as(props)

            values = props * values
            values = values.sum(dim=3)

            # remove the chunk dimensions
            indices, values = indices.view(b, -1 , r), values.view(b, -1)

            if self.templated:
                # stitch the generated indices into the template
                b, l, r = indices.size()
                h, w = self.temp_indices.size()
                template = self.temp_indices[None, :, None, :].expand(b, h, l//h, w)
                template = template.contiguous().view(b, l, w)

                template[:, :, self.learn_cols] = indices
                indices = template

            if self.use_cuda:
                indices = indices.cuda()

        size = self.out_size + input.size()[1:]
        output = tensors.contract(indices, values, size, input)

        if self.bias_type == Bias.DENSE:
            return output + bias
        return output

class NASLayer(SparseLayer):
    """
    Sparse layer with free sparse parameters, no hypernetwork, no template.
    """

    def __init__(self, in_size, out_size, k,
                 sigma_scale=0.2,
                 fix_values=False, has_bias=False,
                 min_sigma=0.0,
                 gadditional=0,
                 rrange=None,
                 radditional=None):

        super().__init__(in_rank=len(in_size),
                         out_size=out_size,
                         bias_type=Bias.DENSE if has_bias else Bias.NONE,
                         gadditional=gadditional,
                         radditional=radditional,
                         rrange=rrange)

        self.k = k
        self.in_size = in_size
        self.out_size = out_size
        self.sigma_scale = sigma_scale
        self.fix_values = fix_values
        self.has_bias = has_bias
        self.min_sigma = min_sigma

        self.rank = len(in_size) + len(out_size)

        self.pmeans = Parameter(torch.randn(k, self.rank))
        self.psigmas = Parameter(torch.randn(k))

        if fix_values:
            self.register_buffer('pvalues', torch.ones(k))
        else:
            self.pvalues = Parameter(torch.randn(k))

        if self.has_bias:
            self.bias = Parameter(torch.randn(*out_size))

    def hyper(self, input, **kwargs):
        """
        Evaluates hypernetwork.
        """

        b = input.size(0)
        size = self.out_size + input.size()[1:] # total dimensions of the weight tensor

        # Expand parameters along batch dimension
        means  = self.pmeans[None, :, :].expand(b, self.k, self.rank)
        sigmas = self.psigmas[None, :].expand(b, self.k)
        values = self.pvalues[None, :].expand(b, self.k)

        means, sigmas = transform_means(means, size), transform_sigmas(sigmas, size)

        if self.has_bias:
            return means, sigmas, values, self.bias
        return means, sigmas, values