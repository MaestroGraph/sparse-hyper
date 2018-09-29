import gaussian
from gaussian import Bias

import torch
from torch.nn import Parameter
from torch import FloatTensor, LongTensor

import abc, itertools, math, types
from numpy import prod

import torch.nn.functional as F


from util import *
import util

import sys
import time, random, logging

from enum import Enum

from tqdm import trange

# added to the sigmas to prevent NaN
EPSILON = 10e-7
SIGMA_BOOST = 2.0

"""
Version of the hyperlayer that uses _global sampling_. It first generates a globakl set of integer tuples and distributes
the weight of all paremeter tuples over these.
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

    batchsize, n, rank = points.size()
    batchsize, k, rank = means.size()
    # batchsize, k, rank = sigmas.size()

    points = points.unsqueeze(2).expand(batchsize, n, k, rank)
    means  = means.unsqueeze(1).expand_as(points)
    sigmas = sigmas.unsqueeze(1).expand_as(points)

    sigmas_squared = torch.sqrt(1.0/(EPSILON+sigmas))

    points = points - means
    points = points * sigmas_squared

    # print(points.size())
    # sys.exit()

    # Compute dot products for all points
    # -- unroll the batch/n dimensions
    points = points.view(-1, 1, rank)
    # -- dot prod
    products = torch.bmm(points, points.transpose(1,2))
    # -- reconstruct shape
    products = products.view(batchsize, n, k)

    num = torch.exp(- 0.5 * products)

    return num

class HyperLayer(nn.Module):

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

    def __init__(self,
                 in_rank, out_shape, additional=0, bias_type=Bias.DENSE, sparse_input=False,
                 subsample=None, relative_range=None, rr_additional=None):
        super().__init__()

        self.use_cuda = False
        self.in_rank = in_rank
        self.out_size = out_shape # without batch dimension
        self.gadditional = additional
        self.region = relative_range
        self.radditional = rr_additional
        self.subsample = subsample

        self.weights_rank = in_rank + len(out_shape) # implied rank of W

        self.bias_type = bias_type
        self.sparse_input = sparse_input
        self.subsample = subsample

        # create a tensor with all binary sequences of length 'rank' as rows
        lsts = [[int(b) for b in bools] for bools in itertools.product([True, False], repeat=self.weights_rank)]
        self.register_buffer('floor_mask', torch.ByteTensor(lsts))

        self.register_buffer('primes', torch.tensor(
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]))

    def bmult(self, width, height, num_indices, batchsize, use_cuda):

        bmult = torch.cuda.LongTensor([height, width]) if use_cuda else LongTensor([height, width])
        m = torch.cuda.LongTensor(range(batchsize)) if use_cuda else LongTensor(range(batchsize))

        bmult = bmult.unsqueeze(0).unsqueeze(0)
        m = m.unsqueeze(1).unsqueeze(1)

        bmult = bmult.expand(batchsize, num_indices, 2)
        m = m.expand(batchsize, num_indices, 2)

        return m * bmult

    def split_out(self, res, input_size, output_size):
        """
        Utility function. res is a B x K x Wrank+2 tensor with range from
        -inf to inf, this function splits out the means, sigmas and values, and
        applies the required activations.

        :param res:
        :param input_size:
        :param output_size:
        :param gain:
        :return:
        """

        b, k, width = res.size()
        w_rank = width - 2

        means = nn.functional.sigmoid(res[:, :, 0:w_rank])
        means = means.unsqueeze(2).contiguous().view(-1, k, w_rank)

        ## expand the indices to the range [0, max]

        # Limits for each of the w_rank indices
        # and scales for the sigmas
        ws = list(output_size) + list(input_size)
        s = torch.cuda.FloatTensor(ws) if self.use_cuda else FloatTensor(ws)
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

    def split_shared(self, res, input_size, output_size, values):
        """
        Splits res into means and sigmas, samples values according to multinomial parameters
        in res

        :param res:
        :param input_size:
        :param output_size:
        :param gain:
        :return:
        """

        b, k, width = res.size()
        w_rank = len(input_size) + len(output_size)

        means = nn.functional.sigmoid(res[:, :, 0:w_rank])
        means = means.unsqueeze(2).contiguous().view(-1, k, w_rank)

        ## expand the indices to the range [0, max]

        # Limits for each of the w_rank indices
        # and scales for the sigmas
        ws = list(output_size) + list(input_size)
        s = torch.cuda.FloatTensor(ws) if self.use_cuda else FloatTensor(ws)
        s = Variable(s.contiguous())

        ss = s.unsqueeze(0).unsqueeze(0)
        sm = s - 1
        sm = sm.unsqueeze(0).unsqueeze(0)

        means = means * sm.expand_as(means)

        sigmas = nn.functional.softplus(res[:, :, w_rank:w_rank+1]) + EPSILON

        sigmas = sigmas.expand_as(means)
        sigmas = sigmas * ss.expand_as(sigmas)

        # extract the values
        vweights = res[:, :, w_rank+1:].contiguous()

        assert vweights.size()[2] == values.size()[0]

        vweights = util.bsoftmax(vweights) + EPSILON

        samples, snode = util.bmultinomial(vweights, num_samples=1)

        weights = values[samples.data.view(-1)].view(b, k)

        return means, sigmas, weights, snode

    def generate_integer_tuples(self, means, rng=None, use_cuda=False, relative_range=None):

        batchsize, n, rank = means.size()

        """
        Generate the neighboring integers
        """

        fm = self.floor_mask.unsqueeze(0).unsqueeze(0).expand(batchsize, n, 2 ** rank, rank)

        neighbor_ints = means.data.unsqueeze(2).expand(batchsize, n, 2 ** rank, rank).contiguous()

        neighbor_ints[fm] = neighbor_ints[fm].floor()
        neighbor_ints[~fm] = neighbor_ints[~fm].ceil()

        neighbor_ints = neighbor_ints.long()

        """
        Sample uniformly from a small range around the given index tuple
        """
        rr_ints = torch.cuda.FloatTensor(batchsize, n, self.radditional, rank) if use_cuda \
            else FloatTensor(batchsize, n, self.radditional, rank)

        rr_ints.uniform_()
        rr_ints *= (1.0 - EPSILON)

        rng = torch.cuda.FloatTensor(rng) if use_cuda else FloatTensor(rng)

        rngxp = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(rr_ints)  # bounds of the tensor
        rrng = torch.cuda.FloatTensor(self.region) if use_cuda else FloatTensor(
            self.region)  # bounds of the range from which to sample
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
        Sample uniformly from all possible index-tuples, with replacement
        """
        sampled_ints = torch.cuda.FloatTensor(batchsize, n, self.gadditional, rank) if use_cuda else FloatTensor(batchsize, n, self.gadditional, rank)

        sampled_ints.uniform_()
        sampled_ints *= (1.0 - EPSILON)

        rngxp = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(sampled_ints)

        sampled_ints = torch.floor(sampled_ints * rngxp).long()

        ints = torch.cat([neighbor_ints, sampled_ints, rr_ints], dim=2)
        # ints = sampled_ints

        # print(ints.size(), ints.view(batchsize, -1, rank).size())

        return ints.view(batchsize, -1, rank)

    def forward(self, input):

        ### Compute and unpack output of hypernetwork

        t0 = time.time()
        bias = None

        if self.bias_type == Bias.NONE:
            means, sigmas, values = self.hyper(input)
        elif self.bias_type == Bias.DENSE:
            means, sigmas, values, bias = self.hyper(input)
        elif self.bias_type == Bias.SPARSE:
            means, sigmas, values, bias_means, bias_sigmas, bias_values = self.hyper(input)
        else:
            raise Exception('bias type {} not recognized.'.format(self.bias_type))

        logging.info('compute hyper: {} seconds'.format(time.time() - t0))

        if self.sparse_input:
            input = input.dense()

        return self.forward_inner(input, means, sigmas, values, bias)

    def forward_inner(self, input, means, sigmas, values, bias):

        t0total = time.time()

        rng = tuple(self.out_size) + tuple(input.size()[1:])

        batchsize = input.size()[0]

        # NB: due to batching, real_indices has shape batchsize x K x rank(W)
        #     real_values has shape batchsize x K

        # turn the real values into integers in a differentiable way
        t0 = time.time()

        if self.subsample is None:
            indices = self.generate_integer_tuples(means, rng=rng, use_cuda=self.use_cuda, relative_range=self.region)
            indfl = indices.float()

            # Mask for duplicate indices
            dups = self.duplicates(indices)

            props = densities(indfl, means, sigmas) # result has size (b, indices.size(1), means.size(1))
            props[dups] == 0
            props = props / props.sum(dim=1, keepdim=True)

            values = values.unsqueeze(1).expand(batchsize, indices.size(1), means.size(1))

            values = props * values
            values = values.sum(dim=2)

        else:
            # For large matrices we need to subsample the means we backpropagate for
            b, nm, rank = means.size()

            sample = random.sample(range(nm), self.subsample) # the means we will learn for
            ids = torch.zeros((nm,), dtype=torch.uint8, device='cuda' if self.use_cuda else 'cpu')
            ids[sample] = 1

            means_in, means_out = means[:, ids, :], means[:, ~ids, :]
            sigmas_in, sigmas_out = sigmas[:, ids, :], sigmas[:, ~ids, :]
            values_in, values_out = values[:, ids], values[:, ~ids]

            indices = self.generate_integer_tuples(means_in, rng=rng, use_cuda=self.use_cuda, relative_range=self.region)
            indfl = indices.float()

            dups = self.duplicates(indices)

            props = densities(indfl, means_in, sigmas_in) # result has size (b, indices.size(1), means.size(1))
            props[dups] == 0
            props = props / props.sum(dim=1, keepdim=True)

            values_in = values_in.unsqueeze(1).expand(batchsize, indices.size(1), means_in.size(1))

            values_in = props * values_in
            values_in = values_in.sum(dim=2)

            means_out = means_out.detach()
            values_out = values_out.detach()

            indices_out = means_out.data.round().long()

            indices = torch.cat([indices, indices_out], dim=1)
            values = torch.cat([values_in, values_out], dim=1)

        # print(values.sum(dim=1))

        # print(indices[0, :])
        # print(dups[0, :])
        # print(values[0, :])
        # sys.exit()

        if self.use_cuda:
            indices = indices.cuda()

        # translate tensor indices to matrix indices so we can use matrix multiplication to perform the tensor contraction
        mindices, flat_size = gaussian.flatten_indices_mat(indices, input.size()[1:], self.out_size)

        ### Create the sparse weight tensor

        x_flat = input.view(batchsize, -1)

        sparsemult = util.sparsemult(self.use_cuda)

        # Prevent segfault
        assert not util.contains_nan(values.data)

        bm = self.bmult(flat_size[1], flat_size[0], mindices.size()[1], batchsize, self.use_cuda)
        bfsize = Variable(flat_size * batchsize)

        bfindices = mindices + bm
        bfindices = bfindices.view(1, -1, 2).squeeze(0)
        vindices = Variable(bfindices.t())

        bfvalues = values.view(1, -1).squeeze(0)
        bfx = x_flat.view(1, -1).squeeze(0)

        # print(vindices.size(), bfvalues.size(), bfsize, bfx.size())
        bfy = sparsemult(vindices, bfvalues, bfsize, bfx)

        y_flat = bfy.unsqueeze(0).view(batchsize, -1)

        y_shape = [batchsize]
        y_shape.extend(self.out_size)

        y = y_flat.view(y_shape) # reshape y into a tensor

        ### Handle the bias
        if self.bias_type == Bias.DENSE:
            y = y + bias
        if self.bias_type == Bias.SPARSE: # untested!
            pass

        return y

    def forward_sample(self, input):
        """
        Samples a single sparse matrix, and computes a transformation with that in a non-differentiable manner.

        :param input:
        :return:
        """

        # Sample k indices

    def backward_sample(self, batch_loss, q_prob, p_prob):
        """
        Computes the gradient by REINFORCE, using the given batch loss, and the probabilities of the sample (as returned by forward_sample)
        :param bacth_loss:
        :param q_prob:
        :param p_prob:
        :return:
        """

class ParamASHLayer(HyperLayer):
    """
    Hyperlayer with free sparse parameters, no hypernetwork (not stricly ASH, should rename).
    """

    def __init__(self, in_shape, out_shape, k, additional=0, sigma_scale=0.2, fix_values=False,  has_bias=False,
                min_sigma=0.0, relative_range=None, rr_additional=None, subsample=None):
        super().__init__(in_rank=len(in_shape), additional=additional, out_shape=out_shape,
                         bias_type=Bias.DENSE if has_bias else Bias.NONE,
                        relative_range=relative_range,
                         rr_additional=rr_additional, subsample=subsample)

        self.k = k
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.sigma_scale = sigma_scale
        self.fix_values = fix_values
        self.has_bias = has_bias
        self.min_sigma = min_sigma

        self.w_rank = len(in_shape) + len(out_shape)

        p = torch.randn(k, self.w_rank + 2)

        # p[:, self.w_rank:self.w_rank + 1] = p[:, self.w_rank:self.w_rank + 1]

        self.params = Parameter(p)

        if self.has_bias:
            self.bias = Parameter(torch.randn(*out_shape))

    def hyper(self, input):
        """
        Evaluates hypernetwork.
        """

        batch_size = input.size()[0]

        # Replicate the parameters along the batch dimension
        res = self.params.unsqueeze(0).expand(batch_size, self.k, self.w_rank+2)

        means, sigmas, values = self.split_out(res, input.size()[1:], self.out_shape)
        sigmas = sigmas * self.sigma_scale + self.min_sigma

        if self.fix_values:
            values = values * 0.0 + 1.0

        if self.has_bias:
            return means, sigmas, values, self.bias

        return means, sigmas, values

    def clone(self):
        result = ParamASHLayer(self.in_shape, self.out_shape, self.k, self.additional, self.gain)

        result.params = Parameter(self.params.data.clone())

        return result

