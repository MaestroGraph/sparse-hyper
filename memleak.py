import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import Parameter
from tqdm import trange

import util, logging, os, psutil

import hyper, gaussian

torch.backends.cudnn.benchmark = True

logging.basicConfig(filename='memleak.log',level=logging.INFO)

torch.manual_seed(2)

CUDA = False

B = 256
M = 32
W, H = 2, 2048

criterion = nn.MSELoss()

class SparseMult(torch.autograd.Function):

    def __init__(self, use_cuda=False):
        super().__init__()
        self.use_cuda = use_cuda

        self.FT =  torch.cuda.sparse.FloatTensor if self.use_cuda else torch.sparse.FloatTensor

    def forward(self, indices, values, size, vector):

        matrix = self.FT(indices, values, torch.Size(size))

        self.save_for_backward(indices, values, size, vector)
        res = torch.mm(matrix, vector.unsqueeze(1))

        return res

    def backward(self, grad_output):

        indices, values, size, vector = self.saved_tensors
        matrix = self.FT(indices, values, torch.Size(size))

        i_ixs = indices[0,:]
        j_ixs = indices[1,:]
        output_select = grad_output.view(-1)[i_ixs]
        vector_select = vector.view(-1)[j_ixs]

        grad_values = output_select *  vector_select

        grad_vector = torch.mm(grad_output.t(), matrix)
        return None, grad_values, None, grad_vector

def iteration():

    sparsemult = util.SparseMult(use_cuda=CUDA)

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

    del sparsemult

for i in trange(int(10e7)):
    iteration()
