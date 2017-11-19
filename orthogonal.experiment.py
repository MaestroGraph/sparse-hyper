import hyper, gaussian
import torch, random, sys
from torch.autograd import Variable
from torch.nn import Parameter
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import util, logging, time, gc

import psutil, os

logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
Learn any orthogonal mapping
"""
w = SummaryWriter()

BATCH = 256
INSHAPE = (4, )
OUTSHAPE = (4, )

CUDA = False

torch.manual_seed(2)

nzs = 4

N = 300000 // BATCH

scale = 2
# plt.figure(figsize=(INSHAPE[0]*scale,OUTSHAPE[0]*scale))
plt.figure(figsize=(5,5))

MARGIN = 0.1
util.makedirs('./spread/')

params = None

model = gaussian.ParamASHLayer(INSHAPE, OUTSHAPE, additional=6, k=nzs, sigma_scale=0.4, fix_values=True)

if CUDA:
    model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for i in trange(N):

    loss, x1, _ = util.orth_loss(BATCH, INSHAPE, model, CUDA)

    loss.backward()        # compute the gradients

    optimizer.step()

    w.add_scalar('orthogonal/loss', loss.data[0], i*BATCH)

    if i % (N//250) == 0:
        means, sigmas, values = model.hyper(x1)

        plt.clf()
        util.plot(means, sigmas, values, shape=(INSHAPE[0], OUTSHAPE[0]))
        plt.xlim((-MARGIN*(INSHAPE[0]-1), (INSHAPE[0]-1) * (1.0+MARGIN)))
        plt.ylim((-MARGIN*(OUTSHAPE[0]-1), (OUTSHAPE[0]-1) * (1.0+MARGIN)))
        plt.savefig('./spread/means{:04}.png'.format(i))

        print('LOSS', torch.sqrt(loss))