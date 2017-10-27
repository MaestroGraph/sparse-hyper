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

"""
Test if we get gradients from a sparse matrix
"""
w = SummaryWriter()

BATCH = 4
SHAPE = (4, )

class SparseMult(torch.autograd.Function):

    def forward(self, indices, values, size, vector):

        matrix = torch.sparse.FloatTensor(indices, values, torch.Size(size))

        self.save_for_backward(indices, values, size, vector)
        res = torch.mm(matrix, vector.unsqueeze(1))
        return res

    def backward(self, grad_output):

        indices, values, size, vector = self.saved_tensors
        matrix = torch.sparse.FloatTensor(indices, values, torch.Size(size))

        i_ixs = indices[0,:]
        j_ixs = indices[1,:]
        output_select = grad_output.view(-1)[i_ixs]
        vector_select = vector.view(-1)[j_ixs]

        grad_values = output_select *  vector_select

        grad_vector = torch.mm(matrix.t(), grad_output).t() \
            if self.needs_input_grad[1] else None

        return None, grad_values, None, grad_vector

class SparseLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.indices = Variable(LongTensor([[0, 1, 2, 0, 2], [0, 1, 2, 3, 3]]))
        self.values = Parameter(FloatTensor([0.1, 0.1, 0.1, 0.1, 0.1]))
        self.size = Variable(torch.LongTensor([3, 4]))

    def forward(self, x):

        return SparseMult()(self.indices, self.values,  self.size, x)

model = SparseLayer()
target = Variable(FloatTensor([[1,0,0,1],[0,1,0,0],[0,0,1,1]]))

N = 200

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for i in trange(N):

    x = Variable(torch.randn((4, )))

    optimizer.zero_grad()

    out = model(x)
    goal = target.mm(x.unsqueeze(1))

    loss = criterion(out, goal) # compute the loss

    loss.backward()             # compute the gradients

    optimizer.step()

    print(model.values)
