import sampling, hyper, gaussian
import torch, random, sys
from torch.autograd import Variable
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import util

torch.manual_seed(1)

"""
Simple experiment: learn the identity function from one tensor to another
"""
w = SummaryWriter()

BATCH = 32
SHAPE = (8, )
CUDA = True

nzs = 8 # hyper.prod(SHAPE)*6

model = gaussian.ParamASHLayer(SHAPE, SHAPE, k=nzs) #
# model.initialize(SHAPE, batch_size=64,iterations=100, lr=0.05)

if CUDA:
    model.cuda()

#x = Variable(torch.rand((BATCH, ) + SHAPE))
#print('--- x', x)

#y = model(x)
#print('--- y', y)

N = 5000

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

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
    means, sigmas, values = model.hyper(x)
    #     print(sigmas, sigmas.grad)
    #     print(values, values.grad)
    #     # print(list(model.parameters()))
    #
    # print('LOSS', torch.sqrt(loss))

    if i % 50 == 0:
        plt.clf()
        util.plot(means, sigmas, values)
        plt.xlim((-1, 8))
        plt.ylim((-1, 8))
        plt.axis('square')
        plt.savefig('./spread/means{:04}.png'.format(i))

for i in range(20):
    x = Variable(torch.rand((1,) + SHAPE))
    y = model(x)

    print('diff', torch.abs(x - y).unsqueeze(0))

    print('********')