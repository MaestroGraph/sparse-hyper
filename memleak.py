import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import Parameter
from tqdm import trange

import util, logging, os, psutil

import hyper, gaussian

logging.basicConfig(filename='memleak.log',level=logging.INFO)

torch.manual_seed(2)

CUDA = True

B = 256
M = 32
W, H = 2, 2048

sparsemult = util.SparseMult(use_cuda=CUDA)
criterion = nn.MSELoss()

def iteration():

    mindices = (torch.rand(B, H, W) * M-1).long()
    values = torch.rand(B, H)
    bsize = LongTensor([M, M])
    y_flat = torch.zeros(B, M)
    x_flat = torch.rand(B, M)
    target = x_flat.clone()

    if CUDA:
        mindices = mindices.cuda()
        bsize = bsize.cuda()
        values = values.cuda()
        x_flat = x_flat.cuda()
        y_flat = y_flat.cuda()
        target = target.cuda()

    bsize = Variable(bsize)
    values = Variable(values, requires_grad=True)
    x_flat = Variable(x_flat)
    y_flat = Variable(y_flat)
    target = Variable(target)

    if CUDA:
        logging.info(util.nvidia_smi())

    for b in range(B):
        bindices = Variable(mindices[b, :, :].squeeze(0).t())
        bvalues = values[b, :]

        bx = x_flat[b, :]

        y_flat[b, :] = sparsemult(bindices, bvalues, bsize, bx)

    loss = criterion(y_flat, target)
    loss.backward(retain_graph=True)

for i in trange(int(10e7)):
    iteration()
