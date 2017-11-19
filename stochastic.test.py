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
Testing stochastic nodes in pytorch

"""
BATCH = 1024
w = SummaryWriter()

target = torch.FloatTensor([[0.1, 0.2, 0.7]])

model = nn.Parameter(torch.rand(1, 3))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([model], lr=0.001)

for i in trange(10000):
    model_sm = nn.Softmax()(model)

    model_sample = torch.multinomial(model_sm.expand(BATCH, 3), 1)
    target_sample = torch.multinomial(target.expand(BATCH, 3), 1)

    reward = (model_sample == Variable(target_sample)).float()

    optimizer.zero_grad()
    model_sample.reinforce(reward.data)
    model_sample.backward()

    optimizer.step()

print(model_sm)

