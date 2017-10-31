import torch
from torch import LongTensor
from torch.autograd import Variable
from torch import nn
from tqdm import trange

import util, logging

torch.backends.cudnn.benchmark = True

torch.manual_seed(2)

CUDA = True
FT = torch.cuda.sparse.FloatTensor if CUDA else torch.sparse.FloatTensor

B = 4
M = 32
W, H = 512, 512

criterion = nn.MSELoss()

class Mult(torch.autograd.Function):

    def __init__(self):
        super().__init__()

    def forward(self, big, vector):

        self.save_for_backward(big)

        return vector * 2.0

    def backward(self, grad_output):

        # indices, values, size, vector = self.saved_tensors
        x = self.saved_tensors

        return None, None, None, None

def iteration():

    mult = Mult()

    big = torch.zeros(B, W, H)
    x = torch.rand(B, M)
    y = torch.zeros(B, M)
    target = x.clone()

    if CUDA:
        big = big.cuda()
        x = x.cuda()
        y = y.cuda()
        target = target.cuda()

    big = Variable(big)
    x = Variable(x, requires_grad=True)
    y = Variable(y)
    target = Variable(target)

    for b in range(B):
        y[b, :] = mult(big[b, :, :], x[b, :])

    loss = criterion(y, target)
    loss.backward()

for i in trange(int(10e7)):
    iteration()
