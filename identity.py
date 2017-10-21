import sampling, hyper, gaussian
import torch, random
from torch.autograd import Variable
from torch import nn, optim
from tqdm import trange
from tensorboardX import SummaryWriter

torch.manual_seed(1)

"""
Simple experiment: learn the identity function from one tensor to another
"""
w = SummaryWriter()

BATCH = 1
SHAPE = (8, )

model = gaussian.DenseASHLayer(SHAPE, SHAPE, k=hyper.prod(SHAPE)*2, hidden=1) #

x = Variable(torch.rand((BATCH, ) + SHAPE))
print('--- x', x)

y = model(x)
print('--- y', y)

N = 50000

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in trange(N):
    x = Variable(torch.rand((BATCH,) + SHAPE))

    optimizer.zero_grad()

    y = model(x)
    loss = criterion(y, x) # compute the loss
    loss.backward()        # compute the gradients
    optimizer.step()

    w.add_scalar('plainvsgauss/rmse', torch.sqrt(loss).data[0], i)

    # if(i != 0 and i % (N/25) == 0):
    #     means, sigmas, values = model.hyper(x)
    #     print(means, means.grad)
    #     print(sigmas, sigmas.grad)
    #     print(values, values.grad)
    #     # print(list(model.parameters()))
    #     print('LOSS', torch.sqrt(loss))

for i in range(20):
    x = Variable(torch.rand((1,) + SHAPE))
    y = model(x)

    print('diff', torch.abs(x - y).unsqueeze(0))

    print('********')