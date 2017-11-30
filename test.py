import torch
from numpy.core.multiarray import dtype
from torch.autograd import Variable

import abc, itertools, math, numpy

from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import torchsample as ts
from torchsample.modules import ModuleTrainer

from torchsample.metrics import *
from tqdm import trange
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import util


mask = torch.ByteTensor([[0, 1],[1, 0]])

vals = torch.FloatTensor([[[0.01, 0.1],[0.2, 0.3]], [[0.01, 0.1],[0.2, 0.3]]])

mask = mask.unsqueeze(0).expand_as(vals)

vals[mask] = vals[mask].ceil()
vals[~mask] = vals[~mask].floor()

print(vals)

# values = Variable(torch.rand(3, 4, 2))
# print(values)
# print(util.bsoftmax(values))

#
# values = Variable(torch.FloatTensor([0.1, 1.1, 2.1]))
# v = Variable(torch.randn(2, 10, 3))
#
# v = nn.functional.softplus(v) + 0.0001
#
# samples = util.bmultinomial(v, 5, replacement=True)
#
# b, r, c = samples.size()
#
# print(values[samples.view(-1)].view(b, r, c))
