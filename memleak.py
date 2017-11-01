import torch
from torch.autograd import Variable
from torch import nn
from tqdm import trange

CUDA = False

B = 2
BIG = 512

criterion = nn.MSELoss()

class Mult(torch.autograd.Function):

    @staticmethod
    def forward(ctx, big, vector):

        # Comment this and problems go away
        ctx.save_for_backward(big)

        return vector * 2.0

    @staticmethod
    def backward(ctx, grad_output):

        return None, None

def iteration():
    mult = Mult.apply

    big = torch.zeros(B, BIG, BIG)
    x = torch.rand(B, 32)
    y = torch.zeros(B, 32)
    target = x.clone()

    if CUDA:
        big, x, y, target = big.cuda(), x.cuda(), y.cuda(), target.cuda()
    big, x, y, target = Variable(big), Variable(x, requires_grad=True), Variable(y), Variable(target)

    for b in range(B):
      y[b, :] = mult(big[b, :, :], x[b, :])

    # Use this instead of the above, and problems go away
    # y = mult(big, x)

    loss = criterion(y, target)

    del mult

for i in trange(int(10e7)):
    iteration()
