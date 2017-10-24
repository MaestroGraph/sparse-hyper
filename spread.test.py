import torch
from numpy.core.multiarray import dtype
from torch.autograd import Variable

import abc, itertools, math, numpy

from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.sparse import FloatTensor

import torchvision
import torchvision.transforms as transforms

import torchsample as ts
from torchsample.modules import ModuleTrainer

from torchsample.metrics import *
from tqdm import trange
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
w = SummaryWriter()

import time, random

n = 5
sub = 20
mid = 0.1

X = Parameter(torch.randn(n, 2))
optimizer = torch.optim.Adam([X])

fig = plt.figure()

for i in trange(1):
    optimizer.zero_grad()

    Xs = X#nn.functional.sigmoid(X)
    r = torch.mm(Xs, Xs.t())
    diag = r.diag().unsqueeze(0).expand_as(r)

    dist = (diag - 2 * r + diag.t())

    loss = (0.0005 * dist ** -2 + dist)
    loss = loss[loss != math.inf]
    print(loss)

    loss = loss.sum() / (2 * n**2)
    print(loss)

    loss.backward()

    print(X.grad)

    optimizer.step()

    w.add_scalar('spacing/loss', loss.data[0], i)

    if i % 200 == 0:
        fig.clear()
        plt.scatter(Xs[:, 0].data.numpy(), Xs[:,1].data.numpy())
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        w.add_image('Scatter', data, i)
        fig.savefig('./anim/frame.{:06}.png'.format(i))

# input = torch.linspace(-10.0, 10.0)
# print(nn.functional.sigmoid(input))

# ind = Variable(torch.rand(5, 2) * 20.0)
# val = Variable(torch.rand(5) * 3.0)
#
# print(ind[1, :], val[1])
#
# ints = Variable( torch.zeros(5, 2 ** len(ind.size()), 2) )
#
# for row in range(ind.size()[0]):
#     for t, bools in enumerate(itertools.product([True, False], repeat=ind.size()[1])):
#
#         for col, bool in enumerate(bools):
#             r = ind[row, col]
#             ints[row, t, col] = torch.floor(r) if bool else torch.ceil(r)
#
# diffs = ints - torch.unsqueeze(ind, 1)
# abs = torch.abs(diffs)
# props = 1.0 - abs
# props = torch.prod(props, 2)
#
# val = torch.unsqueeze(val, 1).expand_as(props).contiguous()
#
# print(props[1, 0].data)
# print(ints[1, 0, :].data)
# print(val[1, 0], val[1,3])
#
# print('***')
#
# ints = ints.view(-1, 2, 1).squeeze()
# props = props.view(-1)
# val = val.view(-1)
#
# print(props[4].data)
# print(ints[4, :].data)
# print(val[4])

# def fi(indices, shape):
#     res = 0
#     for i in range(len(indices)):
#         prod = 1
#         for j in range(i + 1, len(shape)):
#             prod *= shape[j]
#
#         res += prod * indices[i]
#
#     return res
#
# print(fi((0, 0, 0), (2, 2, 2)))
# print(fi((0, 0, 1), (2, 2, 2)))
# print(fi((0, 1, 0), (2, 2, 2)))
# print(fi((0, 1, 1), (2, 2, 2)))
# print(fi((1, 0, 0), (2, 2, 2)))
# print(fi((1, 0, 1), (2, 2, 2)))
# print(fi((1, 1, 0), (2, 2, 2)))
# print(fi((1, 1, 1), (2, 2, 2)))
#
#
# m = torch.LongTensor([[15, 32],
#                       [-6, 3]])
#
# v = m.view(-1)
#
# print(v[fi((0, 0), (2, 2))], m[0, 0])
# print(v[fi((0, 1), (2, 2))], m[0, 1])
# print(v[fi((1, 0), (2, 2))], m[1, 0])
# print(v[fi((1, 1), (2, 2))], m[1, 1])
#
# print(type(m[0, 0]))


