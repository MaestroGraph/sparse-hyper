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
Simple experiment: learn the identity function from one tensor to another
"""
w = SummaryWriter()

BATCH = 256
SHAPE = (32, )
CUDA = True

torch.manual_seed(2)

nzs = hyper.prod(SHAPE)

N = 300000 // BATCH

plt.figure(figsize=(5,5))
util.makedirs('./spread/')

params = None

model = gaussian.ParamASHLayer(SHAPE, SHAPE, additional=256, k=nzs, gain=1.0)
# model.initialize(SHAPE, batch_size=64, iterations=100, lr=0.05)

if CUDA:
    model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
    logging.info('backward: {} seconds'.format(time.time() - t0))

    optimizer.step()

    w.add_scalar('identity32/loss', loss.data[0], i*BATCH)

    if i % (N//50) == 0:
        means, sigmas, values = model.hyper(x)

        plt.clf()
        util.plot(means, sigmas, values)
        plt.xlim((-1, SHAPE[0]))
        plt.ylim((-1, SHAPE[0]))
        plt.savefig('./spread/means{:04}.png'.format(i))

        print('LOSS', torch.sqrt(loss))