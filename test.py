import torch
from numpy.core.multiarray import dtype
from torch.autograd import Variable

import abc, itertools, math

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.sparse import FloatTensor

import torchvision
import torchvision.transforms as transforms

import torchsample as ts
from torchsample.modules import ModuleTrainer

from torchsample.metrics import *

import time

ind = Variable(torch.rand(5, 2) * 20.0)
val = Variable(torch.rand(5) * 3.0)

print(ind[1, :], val[1])

ints = Variable( torch.zeros(5, 2 ** len(ind.size()), 2) )

for row in range(ind.size()[0]):
    for t, bools in enumerate(itertools.product([True, False], repeat=ind.size()[1])):

        for col, bool in enumerate(bools):
            r = ind[row, col]
            ints[row, t, col] = torch.floor(r) if bool else torch.ceil(r)

diffs = ints - torch.unsqueeze(ind, 1)
abs = torch.abs(diffs)
props = 1.0 - abs
props = torch.prod(props, 2)

val = torch.unsqueeze(val, 1).expand_as(props).contiguous()

print(props[1, 0].data)
print(ints[1, 0, :].data)
print(val[1, 0], val[1,3])

print('***')

ints = ints.view(-1, 2, 1).squeeze()
props = props.view(-1)
val = val.view(-1)

print(props[4].data)
print(ints[4, :].data)
print(val[4])

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


