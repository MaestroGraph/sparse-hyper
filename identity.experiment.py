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
SHAPE = (4, )
CUDA = False
MARGIN = 0.1

REPEATS = 3

torch.manual_seed(6)

nzs = hyper.prod(SHAPE)

N = 500 # 64000 // BATCH

plt.figure(figsize=(5,5))
util.makedirs('./spread/')

params = None

for tf in [False, True]:

    sigms = []
    losses = []

    gaussian.PROPER_SAMPLING = tf

    offset = -0.005 if tf else 0.005

    for s in np.linspace(0.1, 0.9, 3):
        for r in trange(REPEATS):

            model = gaussian.ParamASHLayer(SHAPE, SHAPE, additional=4, k=nzs, sigma_scale=s, fix_values=True)

            if CUDA:
                model.cuda()

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.05)

            for i in range(N):

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

                # if False or i % (N//50) == 0:
                #     means, sigmas, values = model.hyper(x)
                #
                #     plt.clf()
                #     util.plot(means, sigmas, values, shape=(SHAPE[0], SHAPE[0]))
                #     plt.xlim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
                #     plt.ylim((-MARGIN*(SHAPE[0]-1), (SHAPE[0]-1) * (1.0+MARGIN)))
                #     plt.savefig('./spread/means{:04}.png'.format(i))

            sigms.append(s + offset)
            losses.append(loss.data[0])

    plt.plot(sigms, losses, marker='.', linewidth=0)

plt.yscale('log')
plt.savefig('losses.pdf')

