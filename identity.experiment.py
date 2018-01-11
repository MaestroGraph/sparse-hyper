import hyper, gaussian
import torch, random, sys
from torch.autograd import Variable
from torch.nn import Parameter
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import util, logging, time, gc
import numpy as np

import psutil, os

logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Simple experiment: learn the identity function from one tensor to another
"""
w = SummaryWriter()

BATCH = 4
SHAPE = (32,)
CUDA = False
MARGIN = 0.1

torch.manual_seed(0)

nzs = hyper.prod(SHAPE)

N = 300000 // BATCH

plt.figure(figsize=(5,5))
util.makedirs('./spread/')

params = None

gaussian.PROPER_SAMPLING = False
model = gaussian.ParamASHLayer(SHAPE, SHAPE, k=32, additional=64, sigma_scale=0.25, has_bias=False)

if CUDA:
    model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for i in trange(N):

    x = torch.rand((BATCH,) + SHAPE)
    if CUDA:
        x = x.cuda()
    x = Variable(x)

    optimizer.zero_grad()

    y = model(x)

    loss = criterion(y, x) # compute the loss

    t0 = time.time()
    loss.backward()        # compute the gradients

    optimizer.step()

    w.add_scalar('identity32/loss', loss.data[0], i*BATCH)

    if False or i % (N//500) == 0:
        means, sigmas, values = model.hyper(x)

        plt.cla()
        util.plot(means, sigmas, values, shape=(SHAPE[0], SHAPE[0]))
        plt.xlim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
        plt.ylim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
        plt.savefig('./identity/means{:04}.png'.format(i))

