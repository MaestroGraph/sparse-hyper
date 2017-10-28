import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import Parameter
from tqdm import trange

import util, logging, os, psutil

logging.basicConfig(filename='memleak.log',level=logging.INFO)

torch.manual_seed(2)

K = 256
S = 64
CUDA = True

class SparseLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.indices = (torch.rand(2, K) * S).long()
        self.values = torch.rand(K)
        self.size = torch.LongTensor([S, S])

        if CUDA:
            self.indices, self.values, self.size = self.indices.cuda(), self.values.cuda(), self.size.cuda()
        self.indices, self.values, self.size = Variable(self.indices), Parameter(self.values), Variable(self.size)

    def forward(self, x):

        return util.SparseMult(use_cuda=CUDA)(self.indices, self.values,  self.size, x)


model = SparseLayer()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for i in trange(int(10e7)):

    x = torch.randn((S,))
    if CUDA:
        x = x.cuda()
    x = Variable(x)

    optimizer.zero_grad()

    out = model(x)

    if i % 1000:
        process = psutil.Process(os.getpid())
        logging.info('{}: memory usage (GB): {}'.format(i, process.memory_info().rss / 10e9))
        logging.info(util.nvidia_smi())

    # loss = criterion(out, goal) # compute the loss
    #
    # loss.backward()             # compute the gradients
    #
    # optimizer.step()
    #
    # print(model.values)
