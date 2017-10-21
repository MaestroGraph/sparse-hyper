import torch
from torch import nn

class Layer(nn.Module):

    def cuda(self, device_id=None):
        print('  CUDA')
        self.use_cuda = True
        super().cuda(device_id)

    def __init__(self):
        super().__init__()

        self.use_cuda = False

class Outer(nn.Module): # Module containing a Layer
    def __init__(self):
        super().__init__()

        self.inner = Layer()

print('sequential model:')

model = nn.Sequential(Layer())
model.cuda() # ! nothing printed

print('module containting Layer:')

model = Outer()
model.cuda() # ! nothing printed

print('calling apply myself:')

model = Outer()
model.apply(lambda t : t.cuda())

print('calling cuda() directly:')

model = Layer()
model.cuda()


