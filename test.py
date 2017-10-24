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
import util

from tensorboardX import SummaryWriter
w = SummaryWriter()

import time, random
import gaussian
BATCH = 1

for i in range(9):
    model = gaussian.DenseASHLayer((8,), (8,), k=32, hidden=1)  #

    x = Variable(torch.rand((BATCH, ) + (8,)))

    means, sigmas, values = model.hyper(x)
    #means = torch.round(means)
    #means = means + Variable(torch.rand(means.size()) * 0.2)

    # print(means)
    plt.figure()
    util.plot(means, sigmas, values)

    plt.xlim((-1, 8))
    plt.ylim((-1, 8))
    plt.axis('equal')

    plt.savefig('means.{}.png'.format(i))



