import sampling, hyper, gaussian
import torch, random, sys
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import Parameter
from tqdm import trange
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import util, logging, time

logging.basicConfig(filename='run.log',level=logging.INFO)

torch.manual_seed(2)

class SparseLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.indices = Variable(LongTensor([[0, 1, 2, 0, 2], [0, 1, 2, 3, 3]]))
        self.values = Parameter(FloatTensor([0.1, 0.1, 0.1, 0.1, 0.1]))
        self.size = Variable(torch.LongTensor([3, 4]))

    def forward(self, x):

        return util.SparseMult()(self.indices, self.values,  self.size, x)


model = SparseLayer()
target = Variable(FloatTensor([[1,0,0,1],[0,1,0,0],[0,0,1,1]]))

N = 200

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for i in trange(N):

    x = Variable(torch.randn((4,)))
    goal = target.mm(x.unsqueeze(1)).t()

    optimizer.zero_grad()

    out = model(x)

    loss = criterion(out, goal) # compute the loss

    loss.backward()             # compute the gradients

    optimizer.step()

    print(model.values)
