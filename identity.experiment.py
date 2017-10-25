import sampling, hyper, gaussian
import torch, random, sys
from torch.autograd import Variable
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import util

torch.manual_seed(2)

"""
Simple experiment: learn the identity function from one tensor to another
"""
w = SummaryWriter()

BATCH = 4
SHAPE = (16, )
CUDA = False

nzs = hyper.prod(SHAPE)

model = gaussian.ParamASHLayer(SHAPE, SHAPE, additional=10, k=nzs, gain=1.0) #
# model.initialize(SHAPE, batch_size=64,iterations=100, lr=0.05)

if CUDA:
    model.cuda()

#x = Variable(torch.rand((BATCH, ) + SHAPE))
#print('--- x', x)

#y = model(x)
#print('--- y', y)

N = 50000 // BATCH

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

plt.figure(figsize=(5,5))
util.makedirs('./spread/')

for i in trange(N):

    x = torch.rand((BATCH,) + SHAPE)
    if CUDA:
        x = x.cuda()
    x = Variable(x)

    optimizer.zero_grad()

    y = model(x)
    loss = criterion(y, x) # compute the loss
    loss.backward()        # compute the gradients
    optimizer.step()

    w.add_scalar('identity/plain', torch.sqrt(loss).data[0], i)

    # if(i != 0 and i % (N/25) == 0):
    #     print(sigmas, sigmas.grad)
    #     print(values, values.grad)
    #     # print(list(model.parameters()))
    #
    # print('LOSS', torch.sqrt(loss))

    if i % (N/50) == 0:
        means, sigmas, values = model.hyper(x)

        plt.clf()
        util.plot(means, sigmas, values)
        plt.xlim((-1, SHAPE[0]))
        plt.ylim((-1, SHAPE[0]))
        plt.savefig('./spread/means{:04}.png'.format(i))

        print(means)
        print(sigmas)
        print(values)
        print('LOSS', torch.sqrt(loss))

for i in range(20):
    x = torch.rand((1,) + SHAPE)
    if CUDA:
        x = x.cuda()
    x = Variable(x)

    y = model(x)

    print('diff', torch.abs(x - y).unsqueeze(0))

    print('********')