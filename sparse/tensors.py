import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable

from sparse.util import prod
import util, sys

"""
Utility functions for manipulation tensors
"""

def flatten_indices_mat(indices, in_shape, out_shape):
    """
    Turns a n NxK matrix of N index-tuples for a tensor T of rank K into an Nx2 matrix M of index-tuples for a _matrix_
    that is created by flattening the first 'in_shape' dimensions into the vertical dimension of M and the remaining
    dimensions in the the horizontal dimension of M.
    :param indices: Long tensor
    :param in_rank:
    :return: (1) A matrix of size N by 2, (2) the dimensions of M
    """

    batchsize, n, rank = indices.size()

    inrank = len(in_shape)
    outrank = len(out_shape)

    result = torch.cuda.LongTensor(batchsize, n, 2) if indices.is_cuda else LongTensor(batchsize, n, 2)

    left = fi_matrix(indices[:, :, 0:outrank], out_shape)   # i index of the weight matrix
    right = fi_matrix(indices[:, :, outrank:rank], in_shape) # j index

    result = torch.cat([left.unsqueeze(2), right.unsqueeze(2)], dim=2)

    return result, LongTensor((prod(out_shape), prod(in_shape)))

def fi_matrix(indices, shape):
    batchsize, rows, rank = indices.size()

    prod = torch.LongTensor(rank).fill_(1)

    if indices.is_cuda:
        prod = prod.cuda()

    for i in range(rank):
        prod[i] = 1
        for j in range(i + 1, len(shape)):
            prod[i] *= shape[j]

    indices = indices * prod.unsqueeze(0).unsqueeze(0).expand_as(indices)

    return indices.sum(dim=2)

def contract(indices, values, size, x, cuda=None):
    """
    Performs a contraction (generalized matrix multiplication) of a sparse tensor with and input x.

    The contraction is defined so that every element of the output is the sum of every element of the input multiplied
    once by a unique element from the tensor (that is, like a fully connected neural network layer). See the paper for
    details.

    :param indices:
    :param values:
    :param size:
    :param x:
    :return:
    """
    # translate tensor indices to matrix indices
    if cuda is None:
        cuda = indices.is_cuda

    b, k, r = indices.size()

    # size is equal to out_size + x.size()
    in_size = x.size()[1:]
    out_size = size[:-len(in_size)]

    assert len(out_size) + len(in_size) == r

    # Flatten into a matrix multiplication
    mindices, flat_size = flatten_indices_mat(indices, x.size()[1:], out_size)
    x_flat = x.view(b, -1, 1)

    # Prevent segfault
    assert mindices.min() >= 0, 'negative index in flattened indices: \n {} \n Original indices {}'.format(mindices, indices)
    assert not util.contains_nan(values.data), 'NaN in values:\n {}'.format(values)

    y_flat = batchmm(mindices, values, flat_size, x_flat, cuda)

    return y_flat.view(b, *out_size)  # reshape y into a tensor


def sparsemm(use_cuda):
    """
    :param use_cuda:
    :return:
    """
    return SparseMMGPU.apply if use_cuda else SparseMMCPU.apply


class SparseMMCPU(torch.autograd.Function):

    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, xmatrix):

        # print(type(size), size, list(size), intlist(size))
        # print(indices.size(), values.size(), torch.Size(intlist(size)))

        matrix = torch.sparse.FloatTensor(indices, values, torch.Size(intlist(size)))

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

class SparseMMGPU(torch.autograd.Function):

    """
    Sparse matrix multiplication with gradients over the value-vector

    Does not work with batch dim.
    """

    @staticmethod
    def forward(ctx, indices, values, size, xmatrix):

        # print(type(size), size, list(size), intlist(size))

        matrix = torch.cuda.sparse.FloatTensor(indices, values, torch.Size(intlist(size)))

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

def batchmm(indices, values, size, xmatrix, cuda=None):
    """
    Multiply a batch of sparse matrices with a batch of dense matrices

    :param indices:
    :param values:
    :param size:
    :param xmatrix:
    :return:
    """

    if cuda is None:
        cuda = indices.is_cuda

    b, n, r = indices.size()
    dv = 'cuda' if cuda else 'cpu'

    height, width = size

    size = torch.tensor(size, device=dv, dtype=torch.long)
    bmult = size[None, None, :].expand(b, n, 2)
    m = torch.arange(b, device=dv, dtype=torch.long)[:, None, None].expand(b, n, 2)

    bindices = (m * bmult).view(b*n, r) + indices.view(b*n, r)

    bfsize = Variable(size * b)
    bvalues = values.contiguous().view(-1)

    b, w, z = xmatrix.size()
    bxmatrix = xmatrix.view(-1, z)

    sm = sparsemm(cuda)
    result = sm(bindices.t(), bvalues, bfsize, bxmatrix)

    return result.view(b, height, -1)

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