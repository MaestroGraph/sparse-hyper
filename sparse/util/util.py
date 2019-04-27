from time import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon, Ellipse, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.axes import Axes
import os, errno, random, time, string, sys

import torch
from torch import nn
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch.utils.data import sampler, dataloader

import torchvision

from collections import OrderedDict

import subprocess

import numpy as np

import math

from enum import Enum

tics = []

DEBUG = False

class Bias(Enum):
    """

    """
    # No bias is used.`c
    NONE = 1

    # The bias is returned as a single dense tensor of floats.
    DENSE = 2

    # The bias is returned in sparse format, in the same way as the weight matrix is.
    SPARSE = 3

def kl_loss(zmean, zlsig):
    """
    Computes the KL loss term for a VAE.

    :param zmean: batch of z means
    :param zlsig: batch of z sigma vectors
    :return:
    """
    b, l = zmean.size()

    kl = 0.5 * torch.sum(zlsig.exp() - zlsig + zmean.pow(2) - 1, dim=1)

    assert kl.size() == (b,)

    return kl

def kl_batch(batch):
    """
    Computes the KL loss between the standard normal MVN and a diagonal MVN fitted to the batch
    :param batch:
    :return:
    """
    b, d = batch.size()

    mean = batch.mean(dim=0, keepdim=True)
    batch = batch - mean

    diacov = torch.bmm(batch.view(d, 1, b), batch.view(d, b, 1)).squeeze() / (b - 1)
    logvar = torch.log(diacov)

    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

def vae_sample(zmean, zlsig, eps=None):
    b, l = zmean.size()

    if eps is None:
        eps = torch.randn(b, l, device='cuda' if zmean.is_cuda else 'cpu')
        eps = Variable(eps)

    return zmean + eps * (zlsig * 0.5).exp()

def tic():
    tics.append(time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time()-tics.pop()


def norm(x):
    """
    Normalize a tensor to a tensor with unit norm (treating first dim as batch dim)

    :param x:
    :return:
    """
    b = x.size()[0]

    n = torch.norm(x.view(b, -1), p=2, dim=1)
    while len(n.size()) < len(x.size()):
        n = n.unsqueeze(1)

    n.expand_as(x)

    return x/n

def makedirs(directory):
    """
    Ensure that all directories in the given path exist.

    :param directory:
    """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def sample(collection, k, required):
    """
    Sample, without replacement, k elements from 'collection', ensuring that 'required' are always contained in the
    sample (but never twice).

    currently only works if collection and required contain only unique elements
    :param k:
    :param collection:
    :param required:
    :return:
    """
    if(k + len(required) > len(collection)):
        # use rejection sampling
        sample = list(collection)
        while len(sample) > k:
            ri = random.choice(range(len(sample)))

            if sample[ri] not in required:
                del(sample[ri])

        return sample
    else:
        required = set(required)
        sample0 = set(random.sample(collection, k + len(required)))
        sample = list(sample0 - required)

        while len(sample) > k - len(required):
            ri = random.choice(range(len(sample)))
            del(sample[ri])

        sample.extend(required)

        return sample
#
# if __name__ == '__main__':
#
#     print('.')
#     print(sample(range(6), 5, [0, 1, 2]))
#     print('.')
#     print(sample(range(100), 6, [0, 1, 2]))
#     print(sample(range(100), 6, [0, 1, 2]))
#     print(sample(range(100), 6, [0, 1, 2]))
#     print('.')

def sparsemult(use_cuda):
    return SparseMultGPU.apply if use_cuda else SparseMultCPU.apply

class SparseMultCPU(torch.autograd.Function):

    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, vector):

        # print(type(size), size, list(size), intlist(size))
        # print(indices.size(), values.size(), torch.Size(intlist(size)))

        matrix = torch.sparse.FloatTensor(indices, values, torch.Size(intlist(size)))

        ctx.indices, ctx.matrix, ctx.vector = indices, matrix, vector

        return torch.mm(matrix, vector.unsqueeze(1))

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data

        # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices

        i_ixs = ctx.indices[0,:]
        j_ixs = ctx.indices[1,:]
        output_select = grad_output.view(-1)[i_ixs]
        vector_select = ctx.vector.view(-1)[j_ixs]

        grad_values = output_select *  vector_select

        grad_vector = torch.mm(ctx.matrix.t(), grad_output).t()
        return None, Variable(grad_values), None, Variable(grad_vector)

class SparseMultGPU(torch.autograd.Function):

    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, vector):

        # print(type(size), size, list(size), intlist(size))

        matrix = torch.cuda.sparse.FloatTensor(indices, values, torch.Size(intlist(size)))

        ctx.indices, ctx.matrix, ctx.vector = indices, matrix, vector

        return torch.mm(matrix, vector.unsqueeze(1))

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.data

        # -- this will break recursive autograd, but it's the only way to get grad over sparse matrices

        i_ixs = ctx.indices[0,:]
        j_ixs = ctx.indices[1,:]
        output_select = grad_output.view(-1)[i_ixs]
        vector_select = ctx.vector.view(-1)[j_ixs]

        grad_values = output_select *  vector_select

        grad_vector = torch.mm(ctx.matrix.t(), grad_output).t()
        return None, Variable(grad_values), None, Variable(grad_vector)

def nvidia_smi():
    command = 'nvidia-smi'
    return subprocess.check_output(command, shell=True)

def orth_loss(batch_size, x_size, model, use_cuda):
    """

    :param batch_size:
    :param x_size:
    :param model:
    :param use_cuda:
    :return:
    """

    x_size = (batch_size,) + x_size

    x1o, x2o = torch.randn(x_size), torch.randn(x_size)

    # normalize to unit tensors
    x1o, x2o = norm(x1o), norm(x2o)

    if use_cuda:
        x1o, x2o = x1o.cuda(), x2o.cuda()
    x1o, x2o = Variable(x1o), Variable(x2o)

    y1 = model(x1o)
    y2 = model(x2o)

    x1 = x1o.view(batch_size, 1, -1)
    x2 = x2o.view(batch_size, 1, -1)
    y1 = y1.view(batch_size, 1, -1)
    y2 = y2.view(batch_size, 1, -1)

    print('x1 v y1', x1[0, :], y1[0, ])

    xnorm = torch.bmm(x1, x2.transpose(1, 2))
    ynorm = torch.bmm(y1, y2.transpose(1, 2))

    loss = torch.sum(torch.pow((xnorm - ynorm), 2)) / batch_size

    return loss, x1o, x2o

def bmultinomial(mat, num_samples=1, replacement=False):
    """
    Take multinomial samples from a batch of matrices with multinomial parameters on the
    rows

    :param mat:
    :param num_samples:
    :param replacement:
    :return:
    """

    batches, rows, columns = mat.size()

    mat = mat.view(1, -1, columns).squeeze(0)

    sample = torch.multinomial(mat, num_samples, replacement)

    return sample.view(batches, rows, num_samples), sample

def bsoftmax(input):

    b, r, c = input.size()
    input = input.view(1, -1, c)
    input = nn.functional.softmax(input.squeeze(0)).unsqueeze(0)

    return input.view(b, r, c)

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)
#
# if __name__ == '__main__':
#
#
#     i = torch.LongTensor([[0, 16, 1],
#                           [2, 0, 2]])
#     v = torch.FloatTensor([1, 1, 1])
#
#     matrix = torch.sparse.FloatTensor(i, v, torch.Size((16, 16)))

def od(lst):
    od = OrderedDict()
    for i, elem in enumerate(lst):
        od[str(i)] = elem

    return od

class Lambda(nn.Module):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class Debug(nn.Module):
    def __init__(self, lambd):
        super(Debug, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        self.lambd(x)
        return x

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)

def flatten(input):
    return input.view(input.size(0), -1)

class NoActivation(nn.Module):
    def forward(self, input):
        return input

def prod(tuple):
    result = 1

    for v in tuple:
        result *= v

    return result

def add_noise(input, std=0.1):
    """
    In-place
    :param input:
    :param std:
    :return:
    """

    noise = torch.cuda.FloatTensor(input.size()) if input.is_cuda else FloatTensor(input.size())
    noise.normal_(std=std)

    return input + noise

def corrupt_(input, prop=0.3):
    """
    Sets a random proportion of the input to zero
    :param input:
    :param prop:
    :return:
    """

    t0 = time.time()
    FT = torch.cuda.FloatTensor if input.is_cuda else torch.FloatTensor
    mask = FT(input.size())
    mask.uniform_()

    mask.sub_(prop).ceil_()

    input.mul_(mask)


def rstring(n):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))

def count_params(model):
    sum = 0
    for tensor in model.parameters():
        sum += prod(tensor.size())

    return sum

def logit(x):
    if type(x) == float:
        return math.log(x / (1 - x))
    return torch.log(x/ (1 - x))


def inv(i):
    sc = (i/27) * 0.9999 + 0.00005
    return logit(sc)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset, using a fixed permutation

    initial source: https://github.com/pytorch/vision/issues/168

    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self,  start, num, total, seed = 0):
        self.start = start
        self.num = num

        self.random = random.Random(seed)

        self.l = list(range(total))
        self.random.shuffle(self.l)

    def __iter__(self):

        return iter(self.l[self.start : self.start + self.num])

    def __len__(self):
        return self.num

def bmult(width, height, num_indices, batchsize, use_cuda):
    """
    ?

    :param width:
    :param height:
    :param num_indices:
    :param batchsize:
    :param use_cuda:
    :return:
    """

    bmult = torch.cuda.LongTensor([height, width]) if use_cuda else LongTensor([height, width])
    m = torch.cuda.LongTensor(range(batchsize)) if use_cuda else LongTensor(range(batchsize))

    bmult = bmult.unsqueeze(0).unsqueeze(0)
    m     = m.unsqueeze(1).unsqueeze(1)

    bmult = bmult.expand(batchsize, num_indices, 2)
    m     = m.expand(batchsize, num_indices, 2)

    return m * bmult

def intlist(tensor):
    """
    A slow and stupid way to turn a tensor into an iterable over ints
    :param tensor:
    :return:
    """
    if type(tensor) is list:
        return tensor

    tensor = tensor.squeeze()

    assert len(tensor.size()) == 1

    s = tensor.size()[0]

    l = [None] * s
    for i in range(s):
        l[i] = int(tensor[i])

    return l

def totensor(dataset, batch_size=512, shuffle=True, maxclass=None):
    """
    Takes a dataset and loads the whole thing into a tensor
    :param dataset:
    :return:
    """

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    index = 0
    for i, batch in enumerate(loader):
        batch, labels = batch[0], batch[1]

        if maxclass is not None:
            batch = batch[labels <= maxclass]

        if i == 0:
            size = list(batch.size())
            size[0] = len(dataset)
            result = torch.zeros(*size)

        result[index:index+batch.size(0)] = batch

        index += batch.size(0)

    return result

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view( (input.size(0),) + self.shape)

def normalize(indices, values, size, row=True, cuda=None, epsilon=0.00000001):
    """
    Row or column normalizes a sparse matrix, defined by the given indices and values. Expects a batch dimension.

    :param indices: (b, k, 2) LongTensor of index tuples
    :param values: length-k vector of values
    :param size: dimensions of the matrix
    :param row: If true, we normalize the rows, otherwise the columns
    :return: The normalized values (the indices stay the same)
    """

    if cuda is None:
        cuda = indices.is_cuda

    dv = 'cuda' if cuda else 'cpu'
    spm = sparsemult(cuda)

    b, k, r = indices.size()

    assert r == 2

    # unroll the batch dimension
    # (think if this as putting all the matrices in the batch along the diagonal of one huge matrix)
    ran = torch.arange(b, device=dv).unsqueeze(1).expand(b, 2)
    ran = ran * torch.tensor(size, device=dv).unsqueeze(0).expand(b, 2)

    offset = ran.unsqueeze(1).expand(b, k, 2).contiguous().view(-1, 2)
    indices = indices.view(-1, 2)

    indices = indices + offset
    values = values.view(-1)

    if row:
        ones = torch.ones((b*size[1],), device=dv)
    else:
        ones = torch.ones((b*size[0],), device=dv)
        # transpose the matrix
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)

    sums = spm(indices.t(), values, torch.tensor(size, device=dv)*b, ones)  # row/column sums

    # select the sums corresponding to each index
    div = torch.index_select(sums, 0, indices[:, 0]).squeeze() + epsilon

    return (values/div).view(b, k)

# if __name__ == "__main__":
#     tind = torch.tensor([[[0, 0],[0, 1], [4, 4], [4, 3]], [[0, 1],[1, 0],[0, 2], [2, 0]]])
#     tv = torch.tensor([[0.5, 0.5, 0.4, 0.6], [0.5, 1, 0.5, 1]])
#
#     print(normalize(tind, tv, (5, 5)))
#     print(normalize(tind, tv, (5, 5), row=False))

def duplicates(tuples):
    """
    Takes a tensor of integer tuples, and for each tuple that occurs multiple times marks all but one of the occurences
    as duplicate.

    :param tuples: A (batch, k, r)-tensor of containing a batch of k r-dimensional integer tuples
    :return: A size (batch, k) byte tensor. When used as a mask, this masks out all duplicates.
    """
    dv = 'cuda' if tuples.is_cuda else 'cpu'

    b, k, r = tuples.size()

    unique = nunique(tuples)

    sorted, sort_idx = torch.sort(unique, dim=1)
    _, unsort_idx = torch.sort(sort_idx, dim=1)

    mask = sorted[:, 1:] == sorted[:, :-1]

    mask = torch.cat([torch.zeros(b, 1, dtype=torch.uint8, device=dv), mask], dim=1)

    return torch.gather(mask, 1, unsort_idx)

def nduplicates(tuples):
    """
    Takes a tensor of integer tuples, and for each tuple that occurs multiple times marks all
    but one of the occurrences as duplicate.

    :param tuples: A (..., k, r)-tensor of containing a batch of k r-dimensional integer tuples
    :return: A size (..., k) byte tensor. When used as a mask, this masks out all duplicates.
    """
    init, k, r = tuples.size()[:-2], tuples.size()[-2], tuples.size()[-1]

    tuples = tuples.view(-1, k, r)
    mask = duplicates(tuples)

    return mask.view(*init, k)

def scatter_imgs(latents, images, size=None, ax=None, color=None, alpha=1.0):

    assert(latents.shape[0] == images.shape[0])

    if ax is None:
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1e-7)
        ax.set_ylim(0, 1e-7)

    if color is None:
        color = np.asarray([0.0, 0.0, 0.0, 1.0])
    else:
        color = np.asarray(color)

    # print(color)

    xmn, ymn = np.min(latents, axis=0)
    xmx, ymx = np.max(latents, axis=0)

    oxmn, oxmx = ax.get_xlim()
    oymn, oymx = ax.get_ylim()

    ax.set_xlim(min(oxmn, xmn), max(oxmx, xmx))
    ax.set_ylim(min(oymn, ymn), max(oymx, ymx))

    # print(ax.get_xlim(), ax.get_ylim())

    if size is None:
        size = (xmx - xmn)/np.sqrt(latents.shape[0])
        size *= 0.5

    n, h, w, c = images.shape

    aspect = h/w

    images = images * (1.0 - color[:3])
    images = 1.0 - images

    for i in range(n):
        x, y = latents[i, 0:2]

        im = images[i, :]
        ax.imshow(im, extent=(x, x + size, y, y + size*aspect), alpha=alpha)

    ax.scatter(latents[:, 0], latents[:, 1], linewidth=0, s=2, color=color)

    return ax, size

def linmoid(x, inf_in, up):
    """
    Squeeze the given input into the range (0, up). All points are translated linearly, except those above and below the
    inflection points (on the input range), which are squeezed through a sigmoid function.

    :param input:
    :param inflections:
    :param range:
    :return:
    """

    ilow  = x < inf_in[0]
    ihigh = x > inf_in[1]

    # linear transform
    s = (up - 1)/(inf_in[1] - inf_in[0])
    y = x * s + 0.5 - inf_in[0] * s

    scale = s * 4
    y[ilow]  = torch.sigmoid((x[ilow] - inf_in[0])*scale)
    y[ihigh] = torch.sigmoid((x[ihigh] - inf_in[1])*scale) - 0.5 + (up - 0.5)

    return y

# if __name__ == "__main__":
#     x = torch.linspace(-0.5, 1.5, 1000)
#     y = linmoid(x, inf_in=(0.25, 0.75), up=3)
#
#     plt.scatter(x.numpy(), y.numpy(), s=2)
#     plt.ylim([0, 3])
#
#     clean()
#     plt.savefig('test_linmoid.png')

def split(offset, depth):
    dv = 'cuda' if offset.is_cuda else 'cpu'

    b, n, s = offset.size()
    bn = b*n

    offset = offset.view(bn, s)

    numbuckets = 2 ** depth # number of buckets in the input
    bsize      = s // numbuckets  # size of the output buckets

    lo = torch.arange(numbuckets, device=dv, dtype=torch.long) * bsize # minimum index of each downbucket
    lo = lo[None, :, None].expand(bn, numbuckets, bsize).contiguous().view(bn, -1)
    hi = torch.arange(numbuckets, device=dv, dtype=torch.long) * bsize + bsize//2  # minimum index of each upbucket
    hi = hi[None, :, None].expand(bn, numbuckets, bsize).contiguous().view(bn, -1)

    upchoices   = offset.long()
    downchoices = 1 - upchoices

    numupchoices = upchoices.view(bn, numbuckets, bsize).cumsum(dim=2).view(bn, -1)
    numdownchoices = downchoices.view(bn, numbuckets, bsize).cumsum(dim=2).view(bn, -1)

    result = torch.zeros(bn, s, dtype=torch.long, device=dv)
    # print(result.dtype, upchoices.dtype, hi.dtype, numupchoices.dtype)
    result = result + upchoices * (hi + numupchoices - 1)
    result = result + downchoices * (lo + numdownchoices - 1)

    # If offset is not arranged correctly (equal numbers of ups and downs per bucket)
    # we get a non-permutation. This is fine, but we must clamp the result to make sure the
    # indices are still legal
    result = result.clamp(0, s-1)

    return result.view(b, n, s)

def sample_offsets(batch, num, size, depth, cuda=False):
    dv = 'cuda' if cuda else 'cpu'

    numbuckets = 2 ** depth # number of buckets in the input
    bsize      = size // numbuckets  # size of the input buckets

    ordered = torch.tensor([0,1], dtype=torch.uint8, device=dv)[None, None, None, :, None].expand(batch, num, numbuckets, 2, bsize // 2)
    ordered = ordered.contiguous().view(batch, num, numbuckets, bsize)

    # shuffle the buckets
    ordered = ordered.view(batch * num * numbuckets, bsize)
    ordered = shuffle_rows(ordered)
    ordered = ordered.view(batch, num, numbuckets, bsize)

    return ordered.contiguous().view(batch, num, -1)


shufflecache = {}
cache_size = 500_000

def shuffle_rows(x):

    r, c = x.size()

    if c not in shufflecache:
        cached = torch.zeros(cache_size, c, dtype=torch.long, device='cpu')
        for i in range(cache_size):
            cached[i, :] = torch.randperm(c)
        shufflecache[c] = cached

    cache = shufflecache[c]
    rows = random.sample(range(cache_size), k=r)
    sample = cache[rows, :]

    if x.is_cuda:
        sample = sample.cuda()

    out = x.gather(dim=1, index=sample)

    if x.is_cuda:
        out = out.cuda()

    return out

# def bunique(tuples):
#     """
#     Like unique/2, but for batched tuples.
#
#     :param tuples: A (b, k, d) tensor of a batch of (k, d) matrices containing d dimensional integer tuples
#     :return: A (b, k, d, 1) tensor
#     """
#
#     b, k, d = tuples.size()
#     tuples = tuples.view(b * k, d)
#
#     un = unique(tuples)
#
#     return un.view(b, k)

def nunique(tuples):
    """

    :param tuples: A (..., d) tensor containing d dimensional integer tuples
    :return: A (..., 1) tensor containing a unique single integer for every integer tuple
    """

    init, d = tuples.size()[:-1], tuples.size()[-1]

    tuples = tuples.view(-1, d)

    un = unique(tuples)

    return un.view(*init)

def unique(tuples):
    """
    Takes a (b, s)-matrix and returns a (b, 1)-matrix with a unique integer for each row.

    Uses the cantor tuple function: https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function

    :param tuples: A matrix of size (b, s)
    :return: A matrix of size (b, 1).
    """

    b, s = tuples.size()

    if s == 1:
        return tuples

    if s == 2:
        k1, k2 = tuples[:, 0], tuples[:, 1]

        res = ((k1 + k2) * (k1 + k2 + 1)) / 2 + k2

        return res[:, None]

    sub = unique(tuples[:, 1:])

    res = torch.cat([tuples[:, 0:1], sub], dim=1)

    return unique(res)

def xent(out, tgt):
    """
    Binary cross-entropy. Manual implementation so we get gradient over both inputs

    :param out:
    :param tgt:
    :return:
    """
    assert out.size() == tgt.size()

    out = out.clamp(0, 1)
    tgt = tgt.clamp(0, 1)

    return - tgt * (out + 1e-10).log() - (1.0 - tgt) * (1.0 - out + 1e-10).log()

if __name__ == '__main__':
#
#      size = 8

#    offset = torch.tensor([1, 1, 0, 1, 1, 0, 0, 0]).byte()
#     offset = torch.tensor([[0, 0, 1, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1, 1, 0]]).byte()

    offset = torch.tensor([[0, 1, 1, 0]]).byte()

    indices = split(offset[:, None, :], 0)

    print(indices)
#
# #    print(sample_offsets(3, 4, 16, 3))
# #
# #     print(unique(torch.tensor( [[1,2,3,4],[4,3,2,1],[1,2,3,4]] )))
# #
# #
#     indices = torch.tensor([[[0, 0], [1, 1]], [[0, 1], [1, 0]]])
#     values = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
#     inputs = torch.tensor([[[1.0, 2.0, 2.0, 2.0, 2.0], [3.0, 4.0, 2.0, 2.0, 2.0]], [[1.0, 2.0, 2.0, 2.0, 2.0], [3.0, 4.0, 4.0, 4.0, 4.0]]])
#
#     print(inputs.size())
#
#     print(batchmm(indices, values, (2,2), inputs))



def wrapmod(x, mod):
    neg = x < 0.0

    y = x.fmod(mod)
    y[neg] = mod + y[neg]

    return y

def interpolation_grid(size=(10, 10)):
    """
    Returns an (h, v, 4) grid, where each point produces a weighted combination of the
    four corner points. Taking the convex combination of tensors using these factors, will
    result in a linear interpolation grid.

    Corner points are enumerated in clockwise fashion, starting top left.

    :param size: h, v
    :return:
    """
    h, v = size
    g1, g2 = torch.meshgrid((torch.linspace(0, 1, h), torch.linspace(0, 1, v)))
    g1, g2 = g1[:, :, None], g2[:, :, None]
    p1, p2 = 1.0 - g1, 1.0 - g2

    return torch.cat([p1*p2, p1*g2, g1*g2, g1*p2], dim=2)

def unsqueezen(input, n):
    """
    Adds n singular dimensions at the start of the
    :param input:
    :param n:
    :return:
    """
    for _ in range(n):
        input = input.unsqueeze(0)
    return input
