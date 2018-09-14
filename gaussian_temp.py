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

from gaussian import Bias, fi_matrix, flatten_indices_mat, densities, tup, fi

# added to the sigmas to prevent NaN
EPSILON = 10e-7
PROPER_SAMPLING = False # NB: set to true for very small tranformations.
BATCH_NEIGHBORS = True # Faster way of finding neighbor index-tuples
SIGMA_BOOST = 2.0

"""
Version of the hyperlayer that learns only over a subset of the columns of the matrix. The rest of the matrix is hardwired.

This can, for instance, be used to create a layer that has 3 incoming connections for each node in the output layer, with
the connections to the input nodes beaing learned.
"""

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

    def __init__(self, in_rank, out_size, temp_indices, learn_cols, additional=0, bias_type=Bias.DENSE, sparse_input=False, subsample=None):
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
        self.additional = additional

        self.bias_type = bias_type
        self.sparse_input = sparse_input
        self.subsample = subsample
        self.learn_cols = learn_cols

        # create a tensor with all binary sequences of length 'out_rank' as rows
        # (this will be used to compute the nearby integer-indices of a float-index).
        lsts = [[int(b) for b in bools] for bools in itertools.product([True, False], repeat=len(learn_cols))]
        self.floor_mask = torch.ByteTensor(lsts)

        # template for the index matrix containing the hardwired connections
        # The learned parts can be set to zero; they will be overriden.
        assert temp_indices.size(1) == in_rank + len(out_size)

        self.register_buffer('temp_indices', temp_indices)

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

        batchsize, n, rank = means.size()

        # ints is the same size as ind, but for every index-tuple in ind, we add an extra axis containing the 2^rank
        # integerized index-tuples we can make from that one real-valued index-tuple
        # ints = torch.cuda.FloatTensor(batchsize, n, 2 ** rank + additional, rank) if use_cuda else FloatTensor(batchsize, n, 2 ** rank, rank)
        t0 = time.time()

        if BATCH_NEIGHBORS:
            fm = self.floor_mask.unsqueeze(0).unsqueeze(0).expand(batchsize, n, 2 ** rank, rank)

            neighbor_ints = means.data.unsqueeze(2).expand(batchsize, n, 2 ** rank, rank).contiguous()

            neighbor_ints[fm] = neighbor_ints[fm].floor()
            neighbor_ints[~fm] = neighbor_ints[~fm].ceil()

            neighbor_ints = neighbor_ints.long()

        else:
            neighbor_ints = torch.LongTensor(batchsize, n, 2 ** rank, rank)

            # produce all integer index-tuples that neighbor the means
            for row in range(n):
                for t, bools in enumerate(itertools.product([True, False], repeat=rank)):

                    for col, bool in enumerate(bools):
                        r = means[:, row, col].data
                        neighbor_ints[:, row, t, col] = torch.floor(r) if bool else torch.ceil(r)

        logging.info('  neighbors: {} seconds'.format(time.time() - t0))

        # Sample additional points
        if rng is not None:
            t0 = time.time()
            total = util.prod(rng)

            if PROPER_SAMPLING:

                ints_flat = torch.LongTensor(batchsize, n, 2 ** rank + additional)

                # flatten
                neighbor_ints = fi(neighbor_ints.view(-1, rank), rng, use_cuda=False)
                neighbor_ints = neighbor_ints.unsqueeze(0).view(batchsize, n, 2 ** rank)

                for b in range(batchsize):
                    for m in range(n):
                        sample = util.sample(range(total), additional + 2 ** rank, list(neighbor_ints[b, m, :]))
                        ints_flat[b, m, :] = torch.LongTensor(sample)

                ints = tup(ints_flat.view(-1), rng, use_cuda=False)
                ints = ints.unsqueeze(0).unsqueeze(0).view(batchsize, n, 2 ** rank + additional, rank)
                ints_fl = ints.float().cuda() if use_cuda else ints.float()

            else:

                sampled_ints = torch.cuda.FloatTensor(batchsize, n, additional, rank) if use_cuda else torch.FloatTensor(batchsize, n, additional, rank)

                sampled_ints.uniform_()
                sampled_ints *= (1.0 - EPSILON)

                rng = torch.cuda.FloatTensor(rng) if use_cuda else torch.FloatTensor(rng)
                rng = rng.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(sampled_ints)

                sampled_ints = torch.floor(sampled_ints * rng).long()

                ints = torch.cat((neighbor_ints, sampled_ints), dim=2)

                ints_fl = ints.float()

            logging.info('  sampling: {} seconds'.format(time.time() - t0))

        ints_fl = Variable(ints_fl)  # leaf node in the comp graph, gradients go through values

        t0 = time.time()
        # compute the proportion of the value each integer index tuple receives
        props = densities(ints_fl, means, sigmas)
        # props is batchsize x K x 2^rank+a, giving a weight to each neighboring or sampled integer-index-tuple

        # -- normalize the proportions of the neigh points and the
        sums = torch.sum(props + EPSILON, dim=2, keepdim=True).expand_as(props)
        props = props / sums

        logging.info('  densities: {} seconds'.format(time.time() - t0))
        t0 = time.time()

        # repeat each value 2^rank+A times, so it matches the new indices
        val = torch.unsqueeze(values, 2).expand_as(props).contiguous()

        # 'Unroll' the ints tensor into a long list of integer index tuples (ie. a matrix of n*2^rank by rank for each
        # instance in the batch) ...
        ints = ints.view(batchsize, -1, rank, 1).squeeze(3)

        # ... and reshape the props and vals the same way
        props = props.view(batchsize, -1)
        val = val.view(batchsize, -1)
        logging.info('  reshaping: {} seconds'.format(time.time() - t0))

        return ints, props, val

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

        batchsize = input.size()[0]

        # NB: due to batching, real_indices has shape batchsize x K x rank(W)
        #     real_values has shape batchsize x K

        # print('--------------------------------')
        # for i in range(util.prod(sigmas.size())):
        #     print(sigmas.view(-1)[i].data[0])

        # turn the real values into integers in a differentiable way
        t0 = time.time()

        # max values allowed for each colum in the index matrix
        fullrange = self.out_size + input.size()[1:]
        subrange = [fullrange[r] for r in self.learn_cols]

        if self.subsample is None:
            indices, props, values = self.discretize(means, sigmas, values, rng=subrange, additional=self.additional, use_cuda=self.use_cuda)
            b, l, r = indices.size()

            # pr = indices.view(-1, r)
            # if torch.sum(pr > torch.cuda.LongTensor(subrange).unsqueeze(0).expand_as(pr)) > 0:
            #     for i in range(b*l):
            #         print(pr[i, :])

            h, w = self.temp_indices.size()
            template = self.temp_indices.unsqueeze(0).unsqueeze(2).expand(b, h, 2**r + self.additional, w)
            template = template.contiguous().view(b, l, w)

            template[:, :, self.learn_cols] = indices # will this work?
            indices = template

            values = values * props

        else: # select a small proportion of the indices to learn over
            raise Exception('Not supported yet.')

            # b, k, r = means.size()
            #
            # prop = torch.cuda.FloatTensor([self.subsample]) if self.use_cuda else torch.FloatTensor([self.subsample])
            #
            # selection = None
            # while (selection is None) or (float(selection.sum()) < 1):
            #     selection = torch.bernoulli(prop.expand(k)).byte()
            #
            # mselection = selection.unsqueeze(0).unsqueeze(2).expand_as(means)
            # sselection = selection.unsqueeze(0).unsqueeze(2).expand_as(sigmas)
            # vselection = selection.unsqueeze(0).expand_as(values)
            #
            # means_in, means_out = means[mselection].view(b, -1, r), means[~ mselection].view(b, -1, r)
            # sigmas_in, sigmas_out = sigmas[sselection].view(b, -1, r), sigmas[~ sselection].view(b, -1, r)
            # values_in, values_out = values[vselection].view(b, -1), values[~ vselection].view(b, -1)
            #
            # means_out = means_out.detach()
            # values_out = values_out.detach()
            #
            # indices_in, props, values_in = self.discretize(means_in, sigmas_in, values_in, rng=rng, additional=self.additional, use_cuda=self.use_cuda)
            # values_in = values_in * props
            #
            # indices_out = means_out.data.round().long()
            #
            # indices = torch.cat([indices_in, indices_out], dim=1)
            # values = torch.cat([values_in, values_out], dim=1)

        logging.info('discretize: {} seconds'.format(time.time() - t0))

        if self.use_cuda:
            indices = indices.cuda()

        # translate tensor indices to matrix indices
        t0 = time.time()

        # mindices, flat_size = flatten_indices(indices, input.size()[1:], self.out_shape, self.use_cuda)
        mindices, flat_size = flatten_indices_mat(indices, input.size()[1:], self.out_size)

        logging.info('flatten: {} seconds'.format(time.time() - t0))

        # NB: mindices is not an autograd Variable. The error-signal for the indices passes to the hypernetwork
        #     through 'values', which are a function of both the real_indices and the real_values.

        ### Create the sparse weight tensor

        x_flat = input.view(batchsize, -1)

        sparsemult = util.sparsemult(self.use_cuda)

        t0 = time.time()

        # Prevent segfault
        assert not util.contains_nan(values.data)

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

        # print(vindices.size(), bfvalues.size(), bfsize, bfx.size())

        bfy = sparsemult(vindices, bfvalues, bfsize, bfx)

        y_flat = bfy.unsqueeze(0).view(batchsize, -1)

        logging.info('sparse mult: {} seconds'.format(time.time() - t0))

        y_shape = [batchsize]
        y_shape.extend(self.out_size)

        y = y_flat.view(y_shape) # reshape y into a tensor

        ### Handle the bias
        if self.bias_type == Bias.DENSE:
            y = y + bias
        if self.bias_type == Bias.SPARSE:
            raise Exception('Not implemented yet.')

        logging.info('total: {} seconds'.format(time.time() - t0total))

        return y
