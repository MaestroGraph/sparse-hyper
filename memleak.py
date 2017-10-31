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

B = 256
M = 32
W, H = 2, 2048

criterion = nn.MSELoss()

class SparseMult(torch.autograd.Function):

    def __init__(self):
        super().__init__()

    def forward(self, indices, values, size, vector):

        matrix = FT(indices, values, torch.Size(size))
        res = torch.mm(matrix, vector.unsqueeze(1))

        self.save_for_backward(indices, values, size, vector)

        return res

    def backward(self, grad_output):

        indices, values, size, vector = self.saved_tensors
        return None, None, None, None

def iteration():

    sparsemult = SparseMult()

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

        # matrix = FT(bindices.data, bvalues.data, torch.Size(bsize.data))

        y_flat[b, :] = sparsemult(bindices, bvalues, bsize, bx)


    loss = criterion(y_flat, target)
    loss.backward(retain_graph=False)

    del sparsemult

for i in trange(int(10e7)):
    iteration()
