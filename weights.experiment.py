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
Simple experiment: learn the identity function with even indices negative
"""
w = SummaryWriter()

BATCH = 64
SHAPE = (4, )
CUDA = False
MARGIN = 0.1

torch.manual_seed(2)

nzs = hyper.prod(SHAPE)

N = 1200000 // BATCH

plt.figure(figsize=(5,5))
util.makedirs('./spread/')

params = None

model = gaussian.WeightSharingASHLayer(SHAPE, SHAPE, additional=6, k=nzs, sigma_scale=0.2, num_values=2)
# model.initialize(SHAPE, batch_size=64, iterations=100, lr=0.05)

if CUDA:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def mse_unsummed(output, target):
    errs = (output - target) ** 2
    return torch.sum(errs, dim=1)

target = torch.eye(SHAPE[0])

for i in range(0, SHAPE[0], 2):
    target[i,i] = - target[i,i]

target = target.unsqueeze(0).expand(BATCH, SHAPE[0], SHAPE[0])

for i in trange(N):

    x = torch.rand((BATCH,) + SHAPE)

    y = torch.bmm(target, x.unsqueeze(2)).squeeze(2)

    if CUDA:
        x, y = x.cuda(), y.cuda()


    x = Variable(x)
    y = Variable(y)

    optimizer.zero_grad()

    out = model(x)
    losses = mse_unsummed(out, y)
    loss = torch.sum(losses)

    # reinforce stochastic nodes
    model.call_reinforce(- losses.data)
    loss.backward()        # compute the gradients

    optimizer.step()

    w.add_scalar('weights/loss', loss.data[0], i*BATCH)

    if i % (N//50) == 0:
        means, sigmas, values = model.hyper(x)

        plt.clf()
        util.plot(means, sigmas, values, shape=(SHAPE[0], SHAPE[0]))
        plt.xlim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
        plt.ylim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
        util.clean()
        plt.savefig('./spread/means{:04}.png'.format(i))

        print('LOSS', torch.sqrt(loss))
        print('sources', model.sources)
        print('sources grad', model.sources.grad)


        vweights = nn.functional.softmax(model.params[:5,-2:])

        print('vweights', vweights)
        print('param grad', model.params.grad)

