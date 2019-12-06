from _context import sparse
from sparse import util

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from util import d

# import warnings
# warnings.simplefilter("error")
# warnings.simplefilter("ignore", DeprecationWarning)

# from util import tic, toc

# NB, the enwik8 data contains tokens from 9 to 240
NUM_TOKENS = 256
LOG2E = math.log2(math.e)
MARGIN = 0.1

def sample(lnprobs, temperature=1.0):

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

class MSparseSelfAttention(nn.Module):
    """
    Masked sparse self attention (two degrees of freedom)
    """
    def __init__(self, emb, k, gadditional, radditional, region, heads=8, mask=False, min_sigma=0.05, sigma_scale=1.0):
        """

        :param emb:
        :param k: Number of connections to the input in total
        :param gadditional:
        :param radditional:
        :param region:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb, self.heads, self.mask, self.min_sigma, self.sigma_scale = emb, heads, mask, min_sigma, sigma_scale

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

        self.gadditional = gadditional
        self.radditional = radditional
        self.region = region
        self.k = k

        self.means  = nn.Parameter(torch.randn((k, 2)))
        self.sigmas = nn.Parameter(torch.randn((k, )))
        self.register_buffer('mvalues', torch.ones((k, )))

    def hyper(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region

        # generate the continuous parameters
        means = self.means[None, None, :, :].expand(b, 1, k, 2)
        sigmas = self.sigmas[None, None, :].expand(b, 1, k)
        values = self.mvalues[None, None, :].expand(b, 1, k)

        means = util.flip(means.contiguous())  # flip everything to below the diagonal of the matrix

        s = (t, t)
        means, sigmas = sparse.transform_means(means, s), \
                        sparse.transform_sigmas(sigmas, s, min_sigma=self.min_sigma) * self.sigma_scale

        return means, sigmas, values

    def forward(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region
        s = (t, t)

        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        means, sigmas, mvalues = self.hyper(x)

        # sample integer indices and values
        indices = sparse.ngenerate(means, self.gadditional, self.radditional, rng=(t, t),
                                   relative_range=(self.region, self.region), cuda=x.is_cuda)
        indices = util.flip(indices)

        indfl = indices.float()

        vs = k * (4 + self.radditional + self.gadditional)
        assert indices.size() == (b, 1, vs, 2), f'{indices.size()}, {(b, 1, vs, 2)}'

        # Mask for duplicate indices
        dups = util.nduplicates(indices).to(torch.bool)

        # compute (unnormalized) densities under the given MVNs (proportions)
        props = sparse.densities(indfl, means, sigmas).clone()
        props[dups, :] = 0
        props = props / props.sum(dim=2, keepdim=True)  # normalize over all points of a given index tuple

        # weight the values by the proportions
        weights = mvalues[:, :, None, :].expand_as(props)
        # - add a dim for the MVNs

        weights = props * weights
        weights = weights.sum(dim=3) # - sum out the MVNs

        assert indices.size() == (b, 1, vs, 2), f'{indices.size()}, {(b, 1, vs, 2)}'
        assert weights.size() == (b, 1, vs), f'{weights.size()},  {(b, 1, vs)}'

        # expand for heads, fold heads into batch
        indices = indices[:, None, :, :, :].expand(b, h, 1, vs, 2).contiguous().view(b*h, vs, 2)
        weights = weights[:, None, :, :].expand(b, h, 1, vs).contiguous().view(b*h, vs)

        # compute keys, queries, values
        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)
        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4)) # b*h, t, e
        keys    = keys    / (e ** (1/4))

        # get dot product of queries and keys
        # - this will be a sparse matrix with the indices we've just computed, and values
        #   defined by the dot product

        # select the queries
        indflat = indices.view(b*h*vs, 2)
        ar = torch.arange(b*h, dtype=torch.long, device=d(x))[:, None].expand(b*h, vs).contiguous().view(b*h*vs)
        squeries = queries[ar, indflat[:, 0], :]
        skeys    = keys   [ar, indflat[:, 1], :]

        dot = torch.bmm(squeries[:, None, :], skeys[:, :, None]).view(b*h, vs)
        dot = sparse.logsoftmax(indices, weights * dot, s)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = sparse.batchmm(indices, dot, size=(t, t), xmatrix=values)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return self.unifyheads(out)


class ASH2DSelfAttention(nn.Module):
    """
    Masked sparse self attention. One degree of freedom, the receptive field is adaptive, based on the incoming
    embedding vector, position embedding and coordinate.
    """
    def __init__(self, emb, k, gadditional, radditional, region, heads=8, mask=False, min_sigma=0.05,
                 sigma_scale=0.1, mmult = 1.0):
        """
        :param emb:
        :param k: Number of connections to the input for each output
        :param gadditional:
        :param radditional:
        :param region:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb, self.heads, self.mask, self.min_sigma, self.sigma_scale = emb, heads, mask, min_sigma, sigma_scale
        self.mmult = mmult

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

        self.gadditional = gadditional
        self.radditional = radditional
        self.region = region
        self.k = k

        self.register_buffer('mvalues', torch.ones((k, )))

        # network that generates the coordinates and sigmas
        hidden = emb * 4
        self.toparams = nn.Sequential(
            nn.Linear(emb + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, k * 3) # two means, one sigma
        )

    def hyper(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region

        # Generate coords
        coords = torch.arange(t, dtype=torch.float, device=d(x)) / t
        coords = coords[None, :, None,].expand(b, t, 1)

        input = torch.cat([x, coords], dim=2)
        params = self.toparams(input) # (b, t, k*3)

        assert not util.contains_nan(params),  \
            f'params contain NaN\n intput {input.min()} {input.max()} \n {list(self.toparams.parameters())}'

        # Generate the logits that correspond to the diagonals of the matrix
        diags = torch.arange(t, dtype=torch.float, device=d(x))
        diags = util.inv(diags, mx=t)

        diags = diags[None, :, None, None].expand(b, t, k, 2)

        means =  params[:, :, :k*2].view(b, t, k, 2)
        sigmas = params[:, :, k*2:].view(b, t, k)
        values = self.mvalues[None, None, :].expand(b, t, k)

        means = diags + self.mmult * means
        means = util.flip(means)

        # means = util.flip(means.contiguous())  # flip everything to below the diagonal of the matrix

        s = (t, t)
        means, sigmas = sparse.transform_means(means, s), \
                        sparse.transform_sigmas(sigmas, s, min_sigma=self.min_sigma) * self.sigma_scale

        return means, sigmas, values

    def forward(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region
        s = (t, t)

        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        means, sigmas, mvalues = self.hyper(x)

        # sample integer indices and values
        indices = sparse.ngenerate(means, self.gadditional, self.radditional, rng=(t, t),
                                   relative_range=(self.region, self.region), cuda=x.is_cuda)

        indices = util.flip(indices)

        indfl = indices.float()

        vs = k * (4 + self.radditional + self.gadditional)
        assert indices.size() == (b, t, vs, 2), f'{indices.size()}, {(b, t, vs, 2)}'

        # Mask for duplicate indices
        dups = util.nduplicates(indices).to(torch.bool)

        # compute (unnormalized) densities under the given MVNs (proportions)
        props = sparse.densities(indfl, means, sigmas).clone()
        props[dups, :] = 0
        props = props / props.sum(dim=2, keepdim=True)  # normalize over all points of a given index tuple

        # weight the values by the proportions
        weights = mvalues[:, :, None, :].expand_as(props)
        # - add a dim for the MVNs

        weights = props * weights
        weights = weights.sum(dim=3) # - sum out the MVNs

        assert indices.size() == (b, t, vs, 2), f'{indices.size()}, {(b, t, vs, 2)}'
        assert weights.size() == (b, t, vs), f'{weights.size()},  {(b, t, vs)}'

        # expand for heads, fold heads into batch
        indices = indices[:, None, :, :, :].expand(b, h, t, vs, 2).contiguous().view(b*h, t*vs, 2)
        weights = weights[:, None, :, :].expand(b, h, t, vs).contiguous().view(b*h, t*vs)

        # compute keys, queries, values
        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)
        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4)) # b*h, t, e
        keys    = keys    / (e ** (1/4))

        # get dot product of queries and keys
        # - this will be a sparse matrix with the indices we've just computed, and values
        #   defined by the dot product

        # select the queries
        indflat = indices.view(b*h*t*vs, 2)
        ar = torch.arange(b*h, dtype=torch.long, device=d(x))[:, None].expand(b*h, t*vs).contiguous().view(b*h*t*vs)
        squeries = queries[ar, indflat[:, 0], :]
        skeys    = keys   [ar, indflat[:, 1], :]

        dot = torch.bmm(squeries[:, None, :], skeys[:, :, None]).view(b*h,t*vs)

        #print(f'dot before {dot.min()}, {dot.mean()}, {dot.max()}')
        assert not util.contains_nan(dot), f'dot contains nan (before softmax) {dot.min()}, {dot.mean()}, {dot.max()}'

        #print(f'dot after  {dot.min()}, {dot.mean()}, {dot.max()}\n')
        dot = sparse.logsoftmax(indices, weights * dot, s).exp()
        # - dot now has row-wise self-attention probabilities

        assert not util.contains_nan(dot), f'dot contains nan (after softmax) {dot.min()}, {dot.mean()}, {dot.max()}'

        # apply the self attention to the values
        out = sparse.batchmm(indices, dot, size=(t, t), xmatrix=values)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        out = self.unifyheads(out)

        assert not util.contains_nan(out), f'output contains nan {out}'

        return out


class ASH1DSelfAttention(nn.Module):
    """
    Masked sparse self attention. One degree of freedom, the receptive field is adaptive, based on the incoming
    embedding vector, position embedding and coordinate.
    """
    def __init__(self, emb, k, gadditional, radditional, region, heads=8, mask=False, min_sigma=0.05, sigma_scale=0.1,
                 mmult = 1.0, norm_method='softmax', outputs=-1,  clamp=True):
        """
        :param emb:
        :param k: Number of connections to the input for each output
        :param gadditional:
        :param radditional:
        :param region:
        :param heads:
        :param outputs: The number of units (at the end of the sequence) to compute new vectors for.
        :param mask:
        """

        super().__init__()

        self.emb, self.heads, self.mask, self.min_sigma, self.sigma_scale = emb, heads, mask, min_sigma, sigma_scale
        self.mmult, self.norm_method, self.clamp = mmult, norm_method, clamp

        if clamp:
            self.mmult *= 3.0

        self.outputs = outputs

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

        self.gadditional = gadditional
        self.radditional = radditional
        self.region = region
        self.k = k

        self.register_buffer('mvalues', torch.ones((k, )))

        # network that generates the coordinates and sigmas
        hidden = emb * 4
        self.toparams = nn.Sequential(
            nn.Linear(emb + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, k * 2) # one mean, one sigma
        )

    def hyper(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region

        o = t if self.outputs < -1 else self.outputs

        # Generate coords
        coords = torch.arange(t, dtype=torch.float, device=d(x)) / t
        coords = coords[None, :, None,].expand(b, t, 1)

        input = torch.cat([x, coords], dim=2)
        params = self.toparams(input) # (b, o, k*2)

        assert not util.contains_nan(params),  \
            f'params contain NaN\n intput {input.min()} {input.max()} \n {list(self.toparams.parameters())}'

        # Generate the logits that correspond to the horizontal coordinate of the current word
        diags = torch.arange(t, dtype=torch.float, device=d(x))
        if not self.clamp:
            diags = util.inv(diags, mx=t)

        diags = diags[None, :, None, None].expand(b, t, k, 1)

        means =  params[:, :, :k].view(b, t, k, 1)
        sigmas = params[:, :, k:].view(b, t, k)
        values = self.mvalues[None, None, :].expand(b, t, k)

        means = diags - self.mmult * F.softplus(means)

        s = (t,)
        means, sigmas = sparse.transform_means(means, s, method='clamp' if self.clamp else 'sigmoid'), \
                        sparse.transform_sigmas(sigmas, s, min_sigma=self.min_sigma) * self.sigma_scale

        return means, sigmas, values

    def forward(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region
        s = (t, t)

        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        means, sigmas, mvalues = self.hyper(x)

        # sample integer indices and values
        indices = sparse.ngenerate(means, self.gadditional, self.radditional, rng=(t,),
                                   relative_range=(self.region, ), cuda=x.is_cuda)
        indfl = indices.float()

        vs = k * (2 + self.radditional + self.gadditional)
        assert indices.size() == (b, t, vs, 1), f'{indices.size()}, {(b, t, vs, 1)}'

        m = torch.arange(t, dtype=torch.long, device=d(indices))[None, :, None, None].expand(b, t, vs, k)

        props = sparse.densities(indfl, means, sigmas).clone() # (b, t, vs, k)

        # Mask for duplicate indices
        dups = util.nduplicates(indices).to(torch.bool)

        # compute (unnormalized) densities under the given MVNs (proportions)
        props[dups, :] = 0
        props[indices > m] = 0

        props = props / props.sum(dim=2, keepdim=True)  # normalize over all points of a given index tuple

        # weight the values by the proportions
        weights = mvalues[:, :, None, :].expand_as(props)
        # - add a dim for the MVNs

        weights = props * weights
        weights = weights.sum(dim=3) # - sum out the MVNs

        out = torch.arange(t, device=d(indices))[None, :, None, None].expand(b, t, vs, 1)
        indices = torch.cat([out, indices], dim=3)

        assert indices.size() == (b, t, vs, 2), f'{indices.size()}, {(b, t, vs, 2)}'
        assert weights.size() == (b, t, vs), f'{weights.size()},  {(b, t, vs)}'

        # expand for heads, fold heads into batch
        indices = indices[:, None, :, :, :].expand(b, h, t, vs, 2).contiguous().view(b*h, t*vs, 2)
        weights = weights[:, None, :, :].expand(b, h, t, vs).contiguous().view(b*h, t*vs)

        # compute keys, queries, values
        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)
        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4)) # b*h, t, e
        keys    = keys    / (e ** (1/4))

        # get dot product of queries and keys
        # - this will be a sparse matrix with the indices we've just computed, and values
        #   defined by the dot product

        # select the queries
        indflat = indices.view(b*h*t*vs, 2)
        ar = torch.arange(b*h, dtype=torch.long, device=d(x))[:, None].expand(b*h, t*vs).contiguous().view(b*h*t*vs)
        squeries = queries[ar, indflat[:, 0], :]
        skeys    = keys   [ar, indflat[:, 1], :]

        dot = torch.bmm(squeries[:, None, :], skeys[:, :, None]).view(b*h,t*vs)

        assert not util.contains_inf(dot), f'dot contains inf (before softmax) {dot.min()}, {dot.mean()}, {dot.max()}'
        assert not util.contains_nan(dot), f'dot contains nan (before softmax) {dot.min()}, {dot.mean()}, {dot.max()}'

        if self.norm_method == 'softmax':
            dot = sparse.logsoftmax(indices, weights * dot, s).exp()
        else:
            dot = sparse.simple_normalize(indices, weights * dot, s, method=self.norm_method)
        # - dot now has row-wise self-attention probabilities

        assert not util.contains_inf(dot), f'dot contains inf (after softmax) {dot.min()}, {dot.mean()}, {dot.max()}'
        assert not util.contains_nan(dot), f'dot contains nan (after softmax) {dot.min()}, {dot.mean()}, {dot.max()}'

        # apply the self attention to the values
        out = sparse.batchmm(indices, dot, size=(t, t), xmatrix=values)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        out = self.unifyheads(out)

        assert not util.contains_nan(out), f'output contains nan {out}, dot min/max: {dot.min()}/{dot.max()}'

        return out


class StridedSparseSelfAttention(nn.Module):
    """
    Masked sparse self attention. One degree of freedom, the receptive field is adaptive, based on the incoming
    embedding vector, position embedding and coordinate.
    """
    def __init__(self, emb, k, gadditional, radditional, region, heads=8, stride=32, mask=False, min_sigma=0.05, sigma_scale=0.1,
                 mmult = 1.0, norm_method='softmax',  clamp=True, **kwargs):
        """
        :param emb:
        :param k: Number of connections to the input for each output
        :param gadditional:
        :param radditional:
        :param region:
        :param heads:
        :param outputs: The number of units (at the end of the sequence) to compute new vectors for.
        :param mask:
        """

        super().__init__()

        self.emb, self.heads, self.mask, self.min_sigma, self.sigma_scale = emb, heads, mask, min_sigma, sigma_scale
        self.mmult, self.norm_method, self.clamp = mmult, norm_method, clamp
        self.stride = stride

        if clamp:
            self.mmult *= 3.0

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

        self.gadditional = gadditional
        self.radditional = radditional
        self.region = region
        self.k = k

        self.register_buffer('mvalues', torch.ones((k, )))

        # network that generates the coordinates and sigmas
        hidden = emb * 4
        self.toparams = nn.Sequential(
            nn.Linear(emb + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, k * 2) # one mean, one sigma
        )

    def hyper(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region
        r = self.stride
        s = (t,)

        # Generate input selection
        selection = torch.arange(t//r, dtype=torch.long, device=d(x))
        selection = (selection + 1) * r - 1
        tp = selection.size(0)

        # Generate coords
        coords = torch.arange(tp, dtype=torch.float, device=d(x)) / tp
        coords = coords[None, :, None,].expand(b, tp, 1)

        input = torch.cat([x[:, selection, :], coords], dim=2)
        params = self.toparams(input) # (b, tp, k*2)

        assert not util.contains_nan(params),  \
            f'params contain NaN\n input {input.min()} {input.max()} \n {list(self.toparams.parameters())}'
        assert not util.contains_inf(params),  \
            f'params contain inf\n input {input.min()} {input.max()} \n {list(self.toparams.parameters())}'

        # Generate the logits/coordinates that correspond to the horizontal coordinate of the current word
        diags = selection.to(torch.float)
        if not self.clamp:
            diags = util.inv(diags, mx=t)

        diags = diags[None, :, None, None].expand(b, tp, k, 1)

        means =  params[:, :, :k].view(b, tp, k, 1)
        sigmas = params[:, :, k:].view(b, tp, k)
        values = self.mvalues[None, None, :].expand(b, tp, k) # all ones atm

        means = diags - self.mmult * F.softplus(means)

        means, sigmas = sparse.transform_means(means, s, method='clamp' if self.clamp else 'sigmoid'), \
                        sparse.transform_sigmas(sigmas, s, min_sigma=self.min_sigma) * self.sigma_scale

        return means, sigmas, values

    def forward(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region
        r = self.stride

        # Generate input selection (the fixed output indices, which are 'stride' units apart)
        selection = torch.arange(t//r, dtype=torch.long, device=d(x))
        selection = (selection + 1) * r - 1
        tp = selection.size(0)

        s = (t, t)

        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        means, sigmas, mvalues = self.hyper(x)

        # sample integer indices and values
        indices = sparse.ngenerate(means, self.gadditional, self.radditional, rng=(t,),
                                   relative_range=(self.region, ), cuda=x.is_cuda, epsilon=10e-5)
        indfl = indices.float()

        vs = k * (2 + self.radditional + self.gadditional) # number of sampled integer index tuples
        assert indices.size() == (b, tp, vs, 1), f'{indices.size()}, {(b, tp, vs, 1)}'

        m = selection[None, :, None, None].expand(b, tp, vs, k)

        props = sparse.densities(indfl, means, sigmas).clone() # (b, tp, vs, k)

        # Mask for duplicate indices
        dups = util.nduplicates(indices).to(torch.bool)

        # compute (unnormalized) densities under the given MVNs (proportions)
        props[dups, :] = 0
        props[indices > m] = 0 # mask out any forward connections
        # -- note that while all the continuous index tuples are guaranteed to point backwards, the sampled discrete
        #    index tuples might point forward, so they still need to be zeroed out here.

        props = props / props.sum(dim=2, keepdim=True)  # normalize over all points of a given index tuple

        # weight the values by the proportions
        weights = mvalues[:, :, None, :].expand_as(props)
        # - add a dim for the MVNs

        weights = props * weights
        weights = weights.sum(dim=3) # - sum out the MVNs

        out = selection[None, :, None, None].expand(b, tp, vs, 1) # output indices
        indices = torch.cat([out, indices], dim=3)

        assert indices.size() == (b, tp, vs, 2), f'{indices.size()}, {(b, tp, vs, 2)}'
        assert weights.size() == (b, tp, vs), f'{weights.size()},  {(b, tp, vs)}'

        assert not util.contains_inf(weights), f'weights contains inf (before norm) {weights.min()}, {weights.mean()}, {weights.max()}'
        assert not util.contains_nan(weights), f'weights contains nan (before norm) {weights.min()}, {weights.mean()}, {weights.max()}'

        # expand for heads, fold heads into batch
        indices = indices[:, None, :, :, :].expand(b, h, tp, vs, 2).contiguous().view(b*h, tp*vs, 2)
        weights = weights[:, None, :, :].expand(b, h, tp, vs).contiguous().view(b*h, tp*vs)

        # compute keys, queries, values
        keys    = self.tokeys(x)   .view(b, t, h, e) # note: t not tp, we compute _all_ queries, keys and values
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)
        # - fold heads into the batch dimension
        keys    = keys.transpose(1, 2).contiguous()   .view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values  = values.transpose(1, 2).contiguous() .view(b * h, t, e)
        # -- We could actually select first, and _then_ transform to kqv's. May be better for very large contexts and
        #    small batches

        queries = queries / (e ** (1/4)) # b*h, t, e
        keys    = keys    / (e ** (1/4))

        # get dot product of queries and keys
        # - this will be a sparse matrix with the indices we've just computed, and values
        #   defined by the dot product

        # select the queries
        indflat = indices.view(b*h*tp*vs, 2)
        ar = torch.arange(b*h, dtype=torch.long, device=d(x))[:, None].expand(b*h, tp*vs).contiguous().view(b*h*tp*vs)
        squeries = queries[ar, indflat[:, 0], :]
        skeys    = keys   [ar, indflat[:, 1], :]

        dot = torch.bmm(squeries[:, None, :], skeys[:, :, None]).view(b*h,tp*vs)
        dot_logits = dot.data.clone()

        assert not util.contains_inf(dot), f'dot contains inf (before norm) {dot.min()}, {dot.mean()}, {dot.max()}'
        assert not util.contains_nan(dot), f'dot contains nan (before norm) {dot.min()}, {dot.mean()}, {dot.max()}'

        if self.norm_method == 'softmax':
            dot = sparse.logsoftmax(indices, weights * dot, s).exp()
        else:
            dot = sparse.simple_normalize(indices, weights * dot, s, method=self.norm_method)
        # - dot now has row-wise self-attention probabilities

        assert not util.contains_inf(dot), f'dot contains inf (after norm) {dot.min()}, {dot.mean()}, {dot.max()}'

        try:
            assert not util.contains_nan(dot), f'dot contains nan (after norm) {dot.min()}, {dot.mean()}, {dot.max()}'
        except AssertionError:

            print(dot.sum(dim=1))
            print('\n\n\n')
            for i in range(b*h):
                print(f'*** {i}')
                print(indices[i])
                print(dot_logits[i])
                print((weights * dot_logits)[i])


                print('\n\n\n')

            sys.exit()

        # apply the self attention to the values
        out = sparse.batchmm(indices, dot, size=s, xmatrix=values)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        out = self.unifyheads(out)

        assert not util.contains_nan(out), f'output contains nan {out}, dot min/max: {dot.min()}/{dot.max()}'

        return out

class ConvSelfAttention(nn.Module):
    """
    Self-attention with a hardwired convolutional sparsity pattern. That is, each node depends on the k
    nodes before.

    Wiring is always "causal" (ie. layer only looks into the past).

    Padding is addded to the input to ensure the input and output have the same length.

    """
    def __init__(self, emb, heads=8, norm_method='softmax', k=32, **kwargs):
        """
        :param emb:
        :param k: Number of connections to the input for each output
        :param gadditional:
        :param radditional:
        :param region:
        :param heads:
        :param outputs: The number of units (at the end of the sequence) to compute new vectors for.
        :param mask:
        """

        super().__init__()

        self.emb, self.heads = emb, heads
        self.norm_method = norm_method

        self.tokeys    = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues  = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

        self.k = k

    def forward(self, x):

        b, t, e = x.size()
        h, k = self.heads, self.k

        tp = t + k - 1
        s = (t, tp)

        xp = F.pad(x, [0, 0, k-1, 0, 0, 0]) # zero pad the beginning of x

        assert xp.size() == (b, tp, e)

        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        # compute keys, queries, values (note that the self attention matrix is slightly rectangular)
        queries = self.toqueries(x).view(b, t, h, e)
        keys    = self.tokeys(xp)  .view(b, tp, h, e)
        values  = self.tovalues(xp).view(b, tp, h, e)

        # - fold heads into the batch dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        keys = keys.transpose(1, 2).contiguous().view(b * h, tp, e)
        values = values.transpose(1, 2).contiguous().view(b * h, tp, e)

        queries = queries / (e ** (1/4)) # b*h, t, e
        keys    = keys    / (e ** (1/4))

        # Get dot product of queries and keys
        # - this will be a sparse matrix with the indices we've just computed, and values
        #   defined by the dot product

        # generate the indices (t*k pairs of integers per attention head)
        indices = torch.arange(t, dtype=torch.long, device=d(x))[:, None, None].expand(t, k, 2).contiguous()
        deltas  = torch.arange(k, dtype=torch.long, device=d(x))[None, :, None].expand(t, k, 1)
        indices[:, :, 1:] += deltas
        indices = indices[None, None, :, :, :].expand(b, h, t, k, 2).contiguous()

        indflat = indices.view(b*h*t*k, 2)

        # select the queries and the keys (left and right column of index matrix) and take their dot
        # product (note that they are already scaled)
        ar = torch.arange(b*h, dtype=torch.long, device=d(x))[:, None].expand(b*h, t*k).contiguous().view(b*h*t*k)
        squeries = queries[ar, indflat[:, 0], :]
        skeys    = keys   [ar, indflat[:, 1], :]

        dot = torch.bmm(squeries[:, None, :], skeys[:, :, None]).view(b*h,t*k)
        indices = indices.view(b*h, t*k, 2)

        # assert not util.contains_inf(dot), f'dot contains inf (before softmax) {dot.min()}, {dot.mean()}, {dot.max()}'
        # assert not util.contains_nan(dot), f'dot contains nan (before softmax) {dot.min()}, {dot.mean()}, {dot.max()}'

        if self.norm_method == 'softmax':
            dot = sparse.logsoftmax(indices, dot, s).exp()
        else:
            dot = sparse.simple_normalize(indices, dot, s, method=self.norm_method)
        # - dot now has row-wise self-attention probabilities

        # assert not util.contains_inf(dot), f'dot contains inf (after softmax) {dot.min()}, {dot.mean()}, {dot.max()}'
        # assert not util.contains_nan(dot), f'dot contains nan (after softmax) {dot.min()}, {dot.mean()}, {dot.max()}'

        # apply the self attention to the values
        out = sparse.batchmm(indices, dot, size=s, xmatrix=values)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        out = self.unifyheads(out)

        assert not util.contains_nan(out), f'output contains nan {out}, dot min/max: {dot.min()}/{dot.max()}'

        return out

class SelfAttention(nn.Module):
    """
    Plain, dense self attention
    """

    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys    = keys    / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

        assert not util.contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0, type='dense', oned=True, **kwargs):
        super().__init__()

        if type == 'sparse':
            if mask:
                if oned:
                    self.attention = ASH1DSelfAttention(emb, heads=heads, **kwargs)
                else:
                    self.attention = ASH2DSelfAttention(emb, heads=heads, **kwargs)
            else:
                raise Exception('Not implemented yet')
        elif type == 'strided':
            self.attention = StridedSparseSelfAttention(emb, heads=heads, **kwargs)
        elif type == 'conv':
            self.attention = ConvSelfAttention(emb, heads, **kwargs)
        elif type == 'dense':
            self.attention = SelfAttention(emb, heads=heads, mask=mask)
        elif type == 'mixed':

            layers = []
            for type in kwargs['mixture']:
                if type == 'c':
                    layers.append(ConvSelfAttention(emb, heads, **kwargs))
                elif type == 's':
                    layers.append(StridedSparseSelfAttention(emb, heads=heads, **kwargs))
                else:
                    raise Exception(f'layer type {type} not recognized/')

            self.attention = nn.Sequential(*layers)

        else:
            raise Exception('Not implemented yet')

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        b, t, e = x.size()

        attended = self.attention(x)

        x = self.norm1(attended + x)
        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)
        x = self.do(x)

        return x

class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, sparse=False, **kwargs):
        """

        :param emb:
        :param heads:
        :param depth:
        :param seq_length:
        :param num_tokens:
        :param sparse:
        :param kwargs: Are passed to the sparse self attention
        """

        super().__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, sparse=sparse, **kwargs))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d(x)))[None, :, :].expand(b, t, e)
        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)

    def forward_for_plot(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        means, sigmas, values = [], [], []

        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d(x)))[None, :, :].expand(b, t, e)
        x = tokens + positions

        for tblock in self.tblocks:
            m, s, v = tblock.attention.hyper(x)
            means.append(m)
            sigmas.append(s)
            values.append(v)

            x = tblock(x)

        return means, sigmas, values

def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    From https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py
    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    X = np.fromstring(open(path).read(n_train + n_valid + n_test), dtype=np.uint8)
    trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
    return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

def go(arg):

    if arg.model.startswith('sparse') or arg.model == 'strided':
        util.makedirs('./transformer-plots/')

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    dv = 'cuda' if arg.cuda else 'cpu'

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    # load the data
    data_train, data_val, data_test = enwik8(arg.data)
    data_test = data_test if arg.final else data_val

    # create the model
    if arg.model.startswith('sparse'):
        model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context,
                             num_tokens=NUM_TOKENS, sparse=True, gadditional=arg.gadditional, radditional=arg.radditional,
                             region=arg.region, k=arg.k, min_sigma=arg.min_sigma, sigma_scale=arg.sigma_mult,
                             oned=(arg.model == 'sparse1d'), norm_method=arg.norm_method, clamp=arg.clamp)
    elif arg.model == 'strided':
        model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context,
                             num_tokens=NUM_TOKENS, gadditional=arg.gadditional, radditional=arg.radditional,
                             region=arg.region, k=arg.k, min_sigma=arg.min_sigma, sigma_scale=arg.sigma_mult,
                             norm_method=arg.norm_method, clamp=arg.clamp, stride=arg.stride, type='strided')
    elif arg.model == 'conv':
        model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context, k=arg.kconv,
                             num_tokens=NUM_TOKENS, type='conv', norm_method=arg.norm_method)
    elif arg.model == 'dense':
        model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context,
                             num_tokens=NUM_TOKENS)
    elif arg.model == 'mixed':
        model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context,
                             num_tokens=NUM_TOKENS, gadditional=arg.gadditional, radditional=arg.radditional,
                             region=arg.region, k=arg.k, min_sigma=arg.min_sigma, sigma_scale=arg.sigma_mult,
                             norm_method=arg.norm_method, clamp=arg.clamp, stride=arg.stride, type='mixed',
                             kconv=arg.kconv, mixture=arg.mixture)

    else:
        raise Exception(f'Model name unknown: {arg.model}')

    if arg.cuda:
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

    # training loop
    for i in tqdm.trange(arg.num_batches):

        if arg.lr_warmup > 0 and i < arg.lr_warmup:
            lr = max(  (arg.lr / arg.lr_warmup) * i, 1e-10)
            opt.lr = lr

        opt.zero_grad()

        # sample batches
        starts = torch.randint(size=(arg.batch_size, ), low=0, high=data_train.size(0) - arg.context - 1)
        seqs_source = [data_train[start  :start+arg.context  ] for start in starts]
        seqs_target = [data_train[start+1:start+arg.context+1] for start in starts]
        source = torch.cat([s[None, :] for s in seqs_source ], dim=0).to(torch.long)
        target = torch.cat([s[None, :] for s in seqs_target ], dim=0).to(torch.long)

        if arg.cuda:
            source, target = source.cuda(), target.cuda()

        source, target = Variable(source), Variable(target)

        output = model(source)

        loss = F.nll_loss(output.transpose(2, 1), target, reduction='none')
        loss = loss.mean()

        tbw.add_scalar('transformer/train-loss', float(loss.item()) * LOG2E, i * arg.batch_size)

        assert loss.item() == loss.item(), f'Loss is nan {loss}'

        loss.backward()

        assert not util.contains_nan(model.parameters()), f'Parameters have become NaN {model.parameters()}'
        if arg.cuda and i == 0 : # occasionally print peak GPU memory usage
            print(f'Peak gpu memory use is {torch.cuda.max_memory_cached() / 1e9:.2} Gb')

        # clip gradients
        if arg.gradient_clipping is not None:
            nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

        opt.step()

        if (arg.model.startswith('sparse') or arg.model == 'strided') and arg.plot_every > 0 and i % arg.plot_every == 0:
            shape = (arg.context, arg.context)

            means, sigmas, values = model.forward_for_plot(source)
            for t, (m, s, v) in enumerate(zip(means, sigmas, values)):

                b, c, k, r = m.size()
                m = m.view(b, c*k, r)
                s = s.view(b, c*k, r)
                v = v.reshape(b, c*k)

                plt.figure(figsize=(7, 7))
                plt.cla()

                if arg.model == 'sparse1d':
                    ind = torch.arange(c, dtype=torch.float, device=d(m))[None, :, None].expand(b, c, k).reshape(b, c*k, 1)
                    m = torch.cat([ind, m], dim=2)
                    util.plot1d(m[0].data, s[0].data, v[0].data, shape=shape)
                elif arg.model == 'strided':
                    r = arg.stride
                    ind = torch.arange(c, dtype=torch.float, device=d(m))
                    ind = (ind + 1) * r - 1
                    ind = ind[None, :, None].expand(b, c, k).reshape(b, c*k, 1)
                    m = torch.cat([ind, m], dim=2)
                    util.plot1d(m[0].data, s[0].data, v[0].data, shape=shape)

                else:
                    util.plot(m, s, v, shape=shape)

                plt.xlim((-MARGIN * (shape[0] - 1), (shape[0] - 1) * (1.0 + MARGIN)))
                plt.ylim((-MARGIN * (shape[0] - 1), (shape[0] - 1) * (1.0 + MARGIN)))

                plt.savefig(f'./transformer-plots/means{i:06}.{t}.pdf')

        if i != 0 and (i % arg.test_every == 0 or i == arg.num_batches - 1):

            upto = data_test.size(0) if i == arg.num_batches - 1 else arg.test_subset
            data_sub = data_test[:upto]

            with torch.no_grad():
                bits, tot = 0.0, 0
                batch = []

                for current in range(data_sub.size(0)):

                    fr = max(0, current - arg.context)
                    to = current + 1

                    context = data_sub[fr:to].to(torch.long)
                    if context.size(0) < arg.context + 1:
                        pad = torch.zeros(size=(arg.context + 1 - context.size(0),), dtype=torch.long)
                        context = torch.cat([pad, context], dim=0)

                        assert context.size(0) == arg.context + 1

                    if arg.cuda:
                        context = context.cuda()

                    batch.append(context[None, :])

                    if len(batch) == arg.test_batchsize or current == data_sub.size(0) - 1:

                        b = len(batch)

                        tot += b

                        all = torch.cat(batch, dim=0)
                        source = all[:, :-1]
                        target = all[:, -1]

                        output = model(source)

                        lnprobs = output[torch.arange(b, device=dv), -1, target]
                        log2probs = lnprobs * LOG2E

                        bits += - log2probs.sum()

                        batch = []

                assert tot == data_sub.size(0)

                bits_per_byte = bits / data_sub.size(0)

                print(f'epoch{i}: {bits_per_byte:.4} bits per byte')
                # print(f'epoch{i}: {bits:.4} total bits')

                tbw.add_scalar(f'transformer/eval-loss', bits_per_byte, i * arg.batch_size)

                # Generate from seed
                GENSIZE = 600
                TEMP = 0.5
                seedfr = random.randint(0, data_test.size(0) - arg.context)
                input = data_test[seedfr:seedfr + arg.context].to(torch.long)

                if arg.cuda:
                    input = input.cuda()

                input = Variable(input)

                print('[', end='', flush=True)
                for c in input:
                    print(str(chr(c)), end='', flush=True)
                print(']', end='', flush=True)

                for _ in range(GENSIZE):
                    output = model(input[None, :])
                    c = sample(output[0, -1, :], TEMP)
                    print(str(chr(max(32, c))), end='', flush=True)

                    input = torch.cat([input[1:], c[None]], dim=0)

                print()

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-N", "--num-batches",
                        dest="num_batches",
                        help="Number of batches to train on. Each batch contains randomly sampled subsequences of the data.",
                        default=1_000_000, type=int)

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Which model to train (dense, sparse1d, sparse2d, conv, mixed).",
                        default='dense', type=str)

    parser.add_argument("--mixture",
                        dest="mixture",
                        help="Character string describing the sequence of convotlutions (c) and strided attentions (s).",
                        default='cccs', type=str)

    parser.add_argument("--norm",
                        dest="norm_method",
                        help="How to normalize the attention matrix (softmax, softplus, abs).",
                        default='softmax', type=str)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples per output in the sparse transformer.",
                        default=32, type=int)

    parser.add_argument("--k-conv",
                        dest="kconv",
                        help="Convolution kernel size.",
                        default=3, type=int)

    parser.add_argument("-a", "--gadditional",
                        dest="gadditional",
                        help="Number of additional points sampled globally",
                        default=8, type=int)

    parser.add_argument("-A", "--radditional",
                        dest="radditional",
                        help="Number of additional points sampled locally",
                        default=8, type=int)

    parser.add_argument("-R", "--region",
                        dest="region",
                        help="Size of the (square) region to use for local sampling.",
                        default=8, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data file",
                        default=None)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-S", "--sigma-mult",
                        dest="sigma_mult",
                        help="Sigma multiplier.",
                        default=0.1, type=float)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Minimum value of sigma.",
                        default=0.01, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=70, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-C", "--context", dest="context",
                        help="Length of the sequences extracted from the corpus (and the context used during inference).",
                        default=300, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr of self-attention layers)",
                        default=4, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--stride",
                        dest="stride",
                        help="Stride length for the strided self attention",
                        default=32, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many batches between tests.",
                        default=1000, type=int)

    parser.add_argument("--plot-every",
                        dest="plot_every",
                        help="How many batches between plotting the sparse indices.",
                        default=100, type=int)

    parser.add_argument("--test-subset",
                        dest="test_subset",
                        help="A subset for the validation tests.",
                        default=100000, type=int)

    parser.add_argument("--test-batchsize",
                        dest="test_batchsize",
                        help="Batch size for computing the validation loss.",
                        default=1024, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=5000, type=int)

    parser.add_argument("--clamp", dest="clamp",
                        help="Use the clamp operation to fit the parameters to the space of index tuples.",
                        action="store_true")


    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
