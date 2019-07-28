import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch import FloatTensor, LongTensor

import abc, itertools, math, types
from numpy import prod

import torch.nn.functional as F

import tensors

from sparse import util
from sparse.util import Bias, sparsemult, contains_nan, bmult, nduplicates

import sys
import random

import numpy as np

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
    # rank: dim of points

    # i: number of integer index tuples sampled per chunk
    # k: number of continuous index tuples per chunk
    # c: number of chunks

    c, i, rank = points.size()[-3:]
    c, k, rank = means.size()[-3:]

    pref = points.size()[:-3]
    assert pref == means.size()[:-3]

    points = points.unsqueeze(-2).expand( *(pref + (c, i, k, rank)) )
    means  = means.unsqueeze(-3).expand_as(points)
    sigmas = sigmas.unsqueeze(-3).expand_as(points)

    sigmas_squared = torch.sqrt(1.0/(EPSILON+sigmas))

    points = points - means
    points = points * sigmas_squared

    # Compute dot products for all points
    # -- unroll the pref/c/k/l dimensions
    points = points.view(-1, 1, rank)
    # -- dot prod

    # print(points)
    products = torch.bmm(points, points.transpose(1, 2))
    # -- reconstruct shape
    products = products.view( *(pref + (c, i, k)) )

    num = torch.exp(- 0.5 * products) # the numerator of the Gaussian density

    return num

def transform_means(means, size, clamp=False):
    """
    Transforms raw parameters for the index tuples (with values in (-inf, inf)) into parameters within the bound of the
    dimensions of the tensor.

    In the case of a templated sparse layer, these parameters and the corresponing size tuple deascribe only the learned
    subtensor.

    :param means: (batch, k, rank) tensor of raw parameter values
    :param size: Tuple describing the tensor dimensions.
    :return:
    """

    # Scale to [0, 1]
    if clamp:
        means = means.clamp(0.0, 1.0)
    else:
        means = F.sigmoid(means)

    # Compute upper bounds
    s = torch.tensor(list(size), dtype=torch.float, device='cuda' if means.is_cuda else 'cpu') - 1
    s = util.unsqueezen(s, len(means.size()) - 1)
    s = s.expand_as(means)

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
    ssize = sigmas.size()
    r = len(size)

    # Scale to [0, 1]
    sigmas = F.softplus(sigmas + SIGMA_BOOST) + min_sigma
    # sigmas = sigmas[:, :, None].expand(b, k, r)
    sigmas = sigmas.unsqueeze(-1).expand(*(ssize + (r, )))

    # Compute upper bounds
    s = torch.tensor(list(size), dtype=torch.float, device='cuda' if sigmas.is_cuda else 'cpu')
    s = util.unsqueezen(s, len(sigmas.size()) - 1)
    s = s.expand_as(sigmas)

    return sigmas * s

class SparseLayer(nn.Module):
    """
    Abstract class for the (templated) hyperlayer. Implement by defining a hypernetwork, and returning it from the
    hyper() method. See NASLayer for an implementation without hypernetwork.

    The templated hyperlayer takes certain columns of its index-tuple matrix as fixed (the template), and others as
    learnable. Imagine a neural network layer where the connections to the output nodes are fixed, but the connections to
    the input nodes can be learned.

    For a non-templated hypernetwork (all columns learnable), just leave the template parameters None.
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
                 gadditional=0, radditional=0, region=None,
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
        :param region: Tuple describing the size of the region over which the local additional points are sampled (must
            be smaller than the size of the tensor).
        :param bias_type: The type of bias of the sparse layer (none, dense or sparse).
        :param subsample:
        """

        super().__init__()
        rank = in_rank + len(out_size)

        assert learn_cols is None or len(region) == len(learn_cols), "Region should span as many dimensions as there are learnable columns"

        self.in_rank = in_rank
        self.out_size = out_size # without batch dimension
        self.gadditional = gadditional
        self.radditional = radditional
        self.region = region
        self.chunk_size = chunk_size

        self.bias_type = bias_type
        self.learn_cols = learn_cols if learn_cols is not None else range(rank)

        self.templated = temp_indices is not None

        # create a tensor with all binary sequences of length 'out_rank' as rows
        # (this will be used to compute the nearby integer-indices of a float-index).
        self.register_buffer('floor_mask', floor_mask(len(self.learn_cols)))

        if self.templated:
            # template for the index matrix containing the hardwired connections
            # The learned parts can be set to zero; they will be overriden.
            assert temp_indices.size(1) == in_rank + len(out_size)

            self.register_buffer('temp_indices', temp_indices)

    # def generate_integer_tuples(self, means, rng=None, relative_range=None, seed=None):
    #     """
    #     Takes continuous-valued index tuples, and generates integer-valued index tuples.
    #
    #     The returned matrix of ints is not a Variable (just a plain LongTensor). Autograd of the real valued indices passes
    #     through the values alone, not the integer indices used to instantiate the sparse matrix.
    #
    #     :param ind: A Variable containing a matrix of N by K, where K is the number of indices.
    #     :param val: A Variable containing a vector of length N containing the values corresponding to the given indices
    #     :return: a triple (ints, props, vals). ints is an N*2^K by K matrix representing the N*2^K integer index-tuples that can
    #         be made by flooring or ceiling the indices in 'ind'. 'props' is a vector of length N*2^K, which indicates how
    #         much of the original value each integer index-tuple receives (based on the distance to the real-valued
    #         index-tuple). vals is vector of length N*2^K, containing the value of the corresponding real-valued index-tuple
    #         (ie. vals just repeats each value in the input 'val' 2^K times).
    #     """
    #
    #     b, k, c, rank = means.size()
    #     dv = 'cuda' if self.is_cuda() else 'cpu'
    #     FT = torch.cuda.FloatTensor if self.is_cuda() else torch.FloatTensor
    #
    #     if seed is not None:
    #         torch.manual_seed(seed)
    #
    #     """
    #     Generate neighbor tuples
    #     """
    #     fm = self.floor_mask[None, None, None, :].expand(b, k, c, 2 ** rank, rank)
    #
    #     neighbor_ints = means.data[:, :, :, None, :].expand(b, k, c, 2 ** rank, rank).contiguous()
    #
    #     neighbor_ints[fm] = neighbor_ints[fm].floor()
    #     neighbor_ints[~fm] = neighbor_ints[~fm].ceil()
    #
    #     neighbor_ints = neighbor_ints.long()
    #
    #     """
    #     Sample uniformly from all integer tuples
    #     """
    #
    #     global_ints = FT(b, k, c, self.gadditional, rank)
    #
    #     global_ints.uniform_()
    #     global_ints *= (1.0 - EPSILON)
    #
    #     rng = FT(rng)
    #     rngxp = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(global_ints)
    #
    #     global_ints = torch.floor(global_ints * rngxp).long()
    #
    #     """
    #     Sample uniformly from a small range around the given index tuple
    #     """
    #     local_ints = FT(b, k, c, self.radditional, rank)
    #
    #     local_ints.uniform_()
    #     local_ints *= (1.0 - EPSILON)
    #
    #     rngxp = rng[None, None, None, :].expand_as(local_ints) # bounds of the tensor
    #
    #     rrng = FT(relative_range) # bounds of the range from which to sample
    #     rrng = rrng[None, None, None, :].expand_as(local_ints)
    #
    #     # print(means.size())
    #     mns_expand = means.round().unsqueeze(3).expand_as(local_ints)
    #
    #     # upper and lower bounds
    #     lower = mns_expand - rrng * 0.5
    #     upper = mns_expand + rrng * 0.5
    #
    #     # check for any ranges that are out of bounds
    #     idxs = lower < 0.0
    #     lower[idxs] = 0.0
    #
    #     idxs = upper > rngxp
    #     lower[idxs] = rngxp[idxs] - rrng[idxs]
    #
    #     local_ints = (local_ints * rrng + lower).long()
    #
    #     all = torch.cat([neighbor_ints, global_ints, local_ints] , dim=3)
    #
    #     return all.view(b, k, -1, rank) # combine all indices sampled within a chunk

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, input, mrange=None, seed=None, **kwargs):
        """

        :param input:
        :param mrange: Specifies a subrange of index tuples to compute the gradient over. This is helpful for gradient
        accumulation methods. This doesn;t work together with templating.
        :param seed:
        :param kwargs:
        :return:
        """

        assert mrange is None or not self.templated, "Templating and gradient accumulation do not work together"

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
        dv = 'cuda' if self.is_cuda() else 'cpu'

        # We divide the list of index tuples into 'chunks'. Each chunk represents a kind of context:
        # - duplicate integer index tuples within the chunk are removed
        # - proportions are normalized over all index tuples within the chunk
        # This is useful in the templated setting. If no chunk size is requested, we just add a singleton dimension.
        k = self.chunk_size if self.chunk_size is not None else n      # chunk size
        c = n // k                                                     # number of chunks

        means, sigmas, values = means.view(b, c, k, r), sigmas.view(b, c, k, r), values.view(b, c, k)

        assert b == input.size(0), 'input batch size ({}) should match parameter batch size ({}).'.format(input.size(0), b)

        # max values allowed for each column in the index matrix
        fullrange = self.out_size + input.size()[1:]
        subrange = [fullrange[r] for r in self.learn_cols]  # submatrix for the learnable dimensions

        if not self.training:
            indices = means.view(b, c*k, r).round().long()

        else:
            if mrange is not None: # only compute the gradient for a subset of index tuples
                fr, to = mrange

                # sample = random.sample(range(nm), self.subsample) # the means we will learn for
                ids = torch.zeros((k,), dtype=torch.uint8, device=dv)
                ids[fr:to] = 1

                means, means_out = means[:, :, ids, :], means[:, :, ~ids, :]
                sigmas, sigmas_out = sigmas[:, :, ids, :], sigmas[:, :, ~ids, :]
                values, values_out = values[:, :, ids], values[:, :, ~ids]

                # These should not get a gradient, since their means aren't being sampled for
                # (their gradient will be computed in other passes)
                means_out, sigmas_out, values_out = means_out.detach(), sigmas_out.detach(), values_out.detach()

            indices = generate_integer_tuples(means, self.gadditional, self.radditional, rng=subrange, relative_range=self.region, seed=seed, cuda=self.is_cuda())
            indfl = indices.float()

            # Mask for duplicate indices
            dups = nduplicates(indices)

            # compute (unnormalized) densities under the given MVNs (proportions)
            props = densities(indfl, means, sigmas).clone()  # result has size (b, c, i, k), i = indices[2]
            props[dups, :] = 0
            props = props / props.sum(dim=2, keepdim=True) # normalize over all points of a given index tuple

            # Weight the values by the proportions
            values = values[:, :, None, :].expand_as(props)

            values = props * values
            values = values.sum(dim=3)

            if mrange is not None:
                indices_out = means_out.data.round().long()
                #
                # print(indices.size(), indices_out.size())
                # print(values.size(), values_out.size())
                # sys.exit()

                indices = torch.cat([indices, indices_out], dim=2)
                values = torch.cat([values, values_out], dim=2)

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

            # if self.is_cuda():
            #     indices = indices.cuda()

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
                 region=None,
                 radditional=None,
                 template=None,
                 learn_cols=None,
                 chunk_size=None):
        """

        :param in_size:
        :param out_size:
        :param k:
        :param sigma_scale:
        :param fix_values:
        :param has_bias:
        :param min_sigma:
        :param gadditional:
        :param region:
        :param radditional:
        :param clamp:
        :param template: LongTensor Template for the matrix of index tuples. Learnable columns are updated through backprop
            other values are taken from the template.
        :param learn_cols: tuple of integers. Learnable columns of the template.

        """

        super().__init__(in_rank=len(in_size),
                         out_size=out_size,
                         bias_type=Bias.DENSE if has_bias else Bias.NONE,
                         gadditional=gadditional,
                         radditional=radditional,
                         region=region,
                         temp_indices=template,
                         learn_cols=learn_cols,
                         chunk_size=chunk_size)

        self.k = k
        self.in_size = in_size
        self.out_size = out_size
        self.sigma_scale = sigma_scale
        self.fix_values = fix_values
        self.has_bias = has_bias
        self.min_sigma = min_sigma

        self.rank = len(in_size) + len(out_size)

        imeans = torch.randn(k, self.rank if template is None else len(learn_cols))
        isigmas = torch.randn(k)

        self.pmeans = Parameter(imeans)
        self.psigmas = Parameter(isigmas)

        if fix_values:
            self.register_buffer('pvalues', torch.ones(k))
        else:
            self.pvalues = Parameter(torch.randn(k))

        if self.has_bias:
            self.bias = Parameter(torch.zeros(*out_size))

    def hyper(self, input, **kwargs):
        """
        Evaluates hypernetwork.
        """

        b = input.size(0)
        size = self.out_size + input.size()[1:] # total dimensions of the weight tensor

        if self.learn_cols is not None:
            size = [size[l] for l in self.learn_cols]

        k, r = self.pmeans.size()

        # Expand parameters along batch dimension
        means  = self.pmeans[None, :, :].expand(b, k, r)
        sigmas = self.psigmas[None, :].expand(b, k)
        values = self.pvalues[None, :].expand(b, k)

        means, sigmas = transform_means(means, size), transform_sigmas(sigmas, size, min_sigma=self.min_sigma) * self.sigma_scale

        if self.has_bias:
            return means, sigmas, values, self.bias
        return means, sigmas, values

class Convolution(nn.Module):
    """
    A non-adaptive hyperlayer that mimics a convolution. That is, the basic structure of the layer is a convolution, but
    instead of connecting every input in the patch to every output channel, we connect them sparsely, with parameters
    learned by the hyperlayer.

    The parameters are the same for each instance of the convolution kernel, but they are sampled separately for each.

    The hyperlayer is _templated_ that is, each connection is fixed to one output node. There are k connections per
    output node.

    The stride is always 1, padding is always added to ensure that the output resolution is the same as the input
    resolution.

    """

    def __init__(self, in_size, out_channels, k, kernel_size=3,
                 gadditional=2, radditional=2, rprop=0.2,
                 min_sigma=0.0,
                 sigma_scale=0.1,
                 fix_values=False,
                 has_bias=True):
        """
        :param in_size: Channels and resolution of the input
        :param out_size: Tuple describing the size of the output.
        :param k: Number of points sampled per instance of the kernel.
        :param kernel_size: Size of the (square) kernel.,

        :param gadditional: Number of points to sample globally per index tuple
        :param radditional: Number of points to sample locally per index tuple
        :param rprop: Describes the region over which the local samples are taken, as a proportion of the channels
        :param bias_type: The type of bias of the sparse layer (none, dense or sparse).
        :param subsample:
        """

        super().__init__()

        rank = 6

        self.in_size = in_size
        self.out_size = (out_channels,) + in_size[1:]
        self.kernel_size = kernel_size
        self.gadditional = gadditional
        self.radditional = radditional
        self.region = (max(1, math.floor(rprop * in_size[0])), kernel_size-1, kernel_size-1)

        self.min_sigma = min_sigma
        self.sigma_scale = sigma_scale

        self.has_bias = has_bias

        self.pad = nn.ZeroPad2d(kernel_size // 2)


        self.means = nn.Parameter(torch.randn(out_channels, k, 3))
        self.sigmas = nn.Parameter(torch.randn(out_channels, k))
        self.values = None if fix_values else nn.Parameter(torch.randn(out_channels, k))

        # out_indices = torch.LongTensor(list(np.ndindex( (in_size[1:]) )))
        # self.register_buffer('out_indices', out_indices)

        template = torch.LongTensor(list(np.ndindex( (out_channels, in_size[1], in_size[2]) )))
        assert template.size() == (prod((out_channels, in_size[1], in_size[2])), 3)
        template = F.pad(template, (0, 3))
        self.register_buffer('template', template)

        if self.has_bias:
            self.bias = Parameter(torch.randn(*self.out_size))

    def hyper(self, x):
        """
        Returns the means, sigmas and values for a _single_ kernel. The same kernel is applied at every position (but
        with fresh samples).

        :param x:
        :return:
        """
        b = x.size(0)

        size = (self.in_size[0], self.kernel_size, self.kernel_size)

        o, k, r = self.means.size()

        # Expand parameters along batch dimension
        means = self.means[None, :, :].expand(b, o, k, r)
        sigmas = self.sigmas[None, :].expand(b, o, k)
        values = self.values[None, :].expand(b, o, k)

        means, sigmas = transform_means(means, size), transform_sigmas(sigmas, size, min_sigma=self.min_sigma) * self.sigma_scale

        return means, sigmas, values

    def forward(self, x):
        dv = 'cuda' if self.template.is_cuda else 'cpu'

        # get continuous parameters
        means, sigmas, values = self.hyper(x)

        # zero pad
        x = self.pad(x)

        b, o, k, r = means.size()
        assert sigmas.size() == (b, o, k, r)
        assert values.size() == (b, o, k)

        # number of instances of the convolution kernel
        nk = self.in_size[1] * self.in_size[2]

        # expand for all kernels
        means  = means [:, :, None, :, :].expand(b, o, nk, k, r)
        sigmas = sigmas[:, :, None, :, :].expand(b, o, nk, k, r)
        values = values[:, :, None, :]   .expand(b, o, nk, k)

        if not self.training:
            indices = means.round().long()

            l = k

        else:
            # sample integer index tuples
            # print(means.size())
            indices = ngenerate(means,
                                self.gadditional, self.radditional,
                                relative_range=self.region,
                                rng=(self.in_size[0], self.kernel_size, self.kernel_size),
                                cuda=means.is_cuda)

            # for i in range(indices.contiguous().view(-1, 3).size(0)):
            #     print(indices.contiguous().view(-1, 3)[i, :])
            # sys.exit()

            # print('indices', indices.size())
            indfl = indices.float()

            b, o, nk, l, r = indices.size()
            assert l == k * (2 ** r + self.gadditional + self.radditional)
            assert nk == self.in_size[1] * self.in_size[2]

            # mask for duplicate indices
            dups = nduplicates(indices)

            # compute unnormalized densities (proportions) under the given MVNs
            props = densities(indfl, means, sigmas).clone()  # result has size (..., c, i, k), i = indices[2]
            # print('densities', props.size())
            # print(util.contains_nan(props))

            props[dups, :] = 0
            # print('... ', props.size())

            props = props / props.sum(dim=-2, keepdim=True) # normalize over all points of a given index tuple

            # print(util.contains_nan(props))
            # sys.exit()

            # Weight the values by the proportions
            values = values[:, :, :, None, :].expand_as(props)

            values = props * values
            values = values.sum(dim=4)

        template = self.template[None, :, None, :].expand(b, self.out_size[0]*self.in_size[1]*self.in_size[2], l, 6)
        template = template.view(b, self.out_size[0], nk, l, 6)

        template[:, :, :, :, 3:] = indices

        indices = template.contiguous().view(b, self.out_size[0] * nk * l, 6)
        offsets = indices[:, :, 1:3]

        # for i in range(indices.view(-1, 6).size(0)):
        #     print(indices.view(-1, 6)[i, :], values.view(-1)[i].data)
        # sys.exit()

        indices[:, :, 4:] = indices[:, :, 4:] + offsets

        values = values.contiguous().view(b, self.out_size[0] * nk * l)

        # apply tensor
        size = self.out_size + x.size()[1:]

        assert (indices.view(-1, 6).max(dim=0)[0] >= torch.tensor(size, device=dv)).sum() == 0, "Max values of indices ({}) out of bounds ({})".format(indices.view(-1, 6).max(dim=0)[0], size)

        output = tensors.contract(indices, values, size, x)

        if self.has_bias:
            return output + self.bias
        return output

FLOOR_MASKS = {}
def floor_mask(num_cols, cuda=False):
    if num_cols not in FLOOR_MASKS:
        lsts = [[int(b) for b in bools] for bools in itertools.product([True, False], repeat=num_cols)]
        FLOOR_MASKS[num_cols] = torch.ByteTensor(lsts, device='cpu')

    if cuda:
        return FLOOR_MASKS[num_cols].cuda()
    return FLOOR_MASKS[num_cols]

def generate_integer_tuples(means, gadditional, ladditional, rng=None, relative_range=None, seed=None, cuda=False, fm=None):
    """
    Takes continuous-valued index tuples, and generates integer-valued index tuples.

    The returned matrix of ints is not a Variable (just a plain LongTensor). Autograd of the real valued indices passes
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
    FT = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if seed is not None:
        torch.manual_seed(seed)

    """
    Generate neighbor tuples
    """
    if fm is None:
        fm = floor_mask(rank, cuda)
    fm = fm[None, None, None, :].expand(b, k, c, 2 ** rank, rank)

    neighbor_ints = means.data[:, :, :, None, :].expand(b, k, c, 2 ** rank, rank).contiguous()

    neighbor_ints[fm] = neighbor_ints[fm].floor()
    neighbor_ints[~fm] = neighbor_ints[~fm].ceil()

    neighbor_ints = neighbor_ints.long()

    """
    Sample uniformly from all integer tuples
    """

    global_ints = FT(b, k, c, gadditional, rank)

    global_ints.uniform_()
    global_ints *= (1.0 - EPSILON)

    rng = FT(rng)
    rngxp = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(global_ints)

    global_ints = torch.floor(global_ints * rngxp).long()

    """
    Sample uniformly from a small range around the given index tuple
    """
    local_ints = FT(b, k, c, ladditional, rank)

    local_ints.uniform_()
    local_ints *= (1.0 - EPSILON)

    rngxp = rng[None, None, None, :].expand_as(local_ints) # bounds of the tensor

    rrng = FT(relative_range) # bounds of the range from which to sample
    rrng = rrng[None, None, None, :].expand_as(local_ints)

    # print(means.size())
    mns_expand = means.round().unsqueeze(3).expand_as(local_ints)

    # upper and lower bounds
    lower = mns_expand - rrng * 0.5
    upper = mns_expand + rrng * 0.5

    # check for any ranges that are out of bounds
    idxs = lower < 0.0
    lower[idxs] = 0.0

    idxs = upper > rngxp
    lower[idxs] = rngxp[idxs] - rrng[idxs]

    local_ints = (local_ints * rrng + lower).long()

    all = torch.cat([neighbor_ints, global_ints, local_ints] , dim=3)

    return all.view(b, k, -1, rank) # combine all indices sampled within a chunk


def ngenerate(means, gadditional, ladditional, rng=None, relative_range=None, seed=None, cuda=False, fm=None):
    """
    Takes continuous-valued index tuples, and generates integer-valued index tuples. Works for inputs with an arbitrary
    number of vectors.

    The returned matrix of ints is not a Variable (just a plain LongTensor). Autograd of the real valued indices passes
    through the values alone, not the integer indices used to instantiate the sparse matrix.

    :param ind: A Variable containing a matrix of N by K, where K is the number of indices.
    :param val: A Variable containing a vector of length N containing the values corresponding to the given indices
    :return: a triple (ints, props, vals). ints is an N*2^K by K matrix representing the N*2^K integer index-tuples that can
        be made by flooring or ceiling the indices in 'ind'. 'props' is a vector of length N*2^K, which indicates how
        much of the original value each integer index-tuple receives (based on the distance to the real-valued
        index-tuple). vals is vector of length N*2^K, containing the value of the corresponding real-valued index-tuple
        (ie. vals just repeats each value in the input 'val' 2^K times).
    """

    b = means.size(0)
    k, c, rank = means.size()[-3:]
    pref = means.size()[:-1]

    FT = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if seed is not None:
        torch.manual_seed(seed)

    """
    Generate neighbor tuples
    """
    if fm is None:
        fm = floor_mask(rank, cuda)

    size = pref + (2**rank, rank)
    fm = util.unsqueezen(fm, len(size) - 2).expand(size)

    neighbor_ints = means.data.unsqueeze(-2).expand(*size).contiguous()

    neighbor_ints[fm] = neighbor_ints[fm].floor()
    neighbor_ints[~fm] = neighbor_ints[~fm].ceil()

    neighbor_ints = neighbor_ints.long()
    # print('means     ', means.contiguous().view(-1, rank).max(dim=0)[0])
    # print('neighbors ', neighbor_ints.view(-1, rank).max(dim=0)[0])

    """
    Sample uniformly from all integer tuples
    """
    gsize = pref + (gadditional, rank)
    global_ints = FT(*gsize)

    global_ints.uniform_()
    global_ints *= (1.0 - EPSILON)

    rng = FT(rng)
    rngxp = util.unsqueezen(rng, len(gsize) - 1).expand_as(global_ints)

    global_ints = torch.floor(global_ints * rngxp).long()

    # print('globals ', global_ints.view(-1, rank).max(dim=0)[0])

    """
    Sample uniformly from a small range around the given index tuple
    """
    lsize = pref + (ladditional, rank)
    local_ints = FT(*lsize)

    local_ints.uniform_()
    local_ints *= (1.0 - EPSILON)

    rngxp = util.unsqueezen(rng, len(lsize) - 1).expand_as(local_ints) # bounds of the tensor

    rrng = FT(relative_range) # bounds of the range from which to sample
    rrng = util.unsqueezen(rrng, len(lsize) - 1).expand_as(local_ints)

    # print(means.size())
    mns_expand = means.round().unsqueeze(-2).expand_as(local_ints)


    # upper and lower bounds
    lower = mns_expand - rrng * 0.5
    upper = mns_expand + rrng * 0.5

    # check for any ranges that are out of bounds
    idxs = lower < 0.0
    lower[idxs] = 0.0

    idxs = upper > rngxp
    lower[idxs] = rngxp[idxs] - rrng[idxs]

    local_ints = (local_ints * rrng + lower).long()

    # print('mns_expand ', mns_expand.view(-1, rank).max(dim=0)[0])
    # print('local      ', local_ints.view(-1, rank).max(dim=0)[0])

    all = torch.cat([neighbor_ints, global_ints, local_ints] , dim=-2)

    fsize = pref[:-1] + (-1, rank)
    return all.view(*fsize) # combine all indices sampled within a chunk