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

CUDA = False

B = 256
M = 32
IN = OUT = tuple([M] * 8)
W, H = len(IN) + len(OUT), 2048

for i in trange(int(10e7)):

    x = (torch.randn((B, H, W)) * M).long()

    if CUDA:
        x = x.cuda()

    x = Variable(x)

    x, _ = gaussian.flatten_indices(x, IN, OUT, use_cuda=CUDA)

    if i % 25 == 0:
        logging.info(util.nvidia_smi())


