import sampling, hyper
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

BATCH = 4
SHAPE = (1, 28, 28)

model = nn.Sequential(
    hyper.ConvASHLayer(SHAPE, (4, 8, 8), k=100000),
    nn.ReLU(),
    hyper.ConvASHLayer((4, 8, 8), (8, 4, 4), k = 100000),
    nn.ReLU(),
    hyper.ConvASHLayer((8, 4, 4), (16, 2, 2), k = 100000),
    nn.ReLU(),
    hyper.Flatten(),
    nn.Linear(64, 10),
    nn.Softmax())

## SIMPLE

N = 50000

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for i in trange(N):
    x = Variable(torch.rand((BATCH,) + SHAPE))

    optimizer.zero_grad()

    y = model(x)
    loss = criterion(y, target) # compute the loss
    loss.backward()        # compute the gradients
    optimizer.step()

    w.add_scalar('sampling-loss/rmse', torch.sqrt(loss).data[0], i)

    if(i != 0 and i % (N/25) == 0):
        means, sigmas, values = model.hyper(x)
        print(means)
        print(sigmas)
        print(values)
        # print(list(model.parameters()))
        print('LOSS', torch.sqrt(loss))

for i in range(20):
    x = Variable(torch.rand((1,) + SHAPE))
    y = model(x)

    print('diff', torch.abs(x - y).unsqueeze(0))

    print('********')