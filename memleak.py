import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import Parameter
from tqdm import trange

import util, logging, os, psutil

import hyper

logging.basicConfig(filename='memleak.log',level=logging.INFO)

torch.manual_seed(2)

B = 256
M = 32
IN = OUT = tuple([M] * 8)
W, H = len(IN) + len(OUT), 2048

for i in trange(int(10e7)):

    x = torch.randn((B, H, W)) * M
    x = x.long().cuda()

    x = Variable(x)

    x, _ = hyper.flatten_indices(x, IN, OUT)

    if i % 25 == 0:
        logging.info(util.nvidia_smi())


