import torch
from torch import nn
from torch.autograd import Variable
import math

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)

class ResBlock(nn.Module):
    """
    Inverted residual block (as in mobilenetv2)
    """
    def __init__(self,  c, wide=None, kernel=3, grouped=True):
        super().__init__()

        wide = 6 * c if wide is None else wide
        padding = int(math.floor(kernel/2))

        self.convs = nn.Sequential(
            nn.Conv2d(c, wide, kernel_size=1),
            nn.Conv2d(wide, wide, kernel_size=kernel, padding=padding, groups=wide if grouped else 1),
            nn.Conv2d(wide, c, kernel_size=1),
            nn.BatchNorm2d(c), nn.ReLU()
        )

    def forward(self, x):

        return self.convs(x) + x # wo the skip, the sefgfault happens immediately


model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            ResBlock(32, grouped=True),  nn.Conv2d(32, 16, kernel_size=1),
            ResBlock(16, grouped=False), nn.Conv2d(16, 16, kernel_size=1),
            nn.MaxPool2d(kernel_size=16),
            Flatten(),
            nn.Linear(16, 10),
            nn.Softmax(dim=-1)
        )


opt = torch.optim.SGD(lr=0.000001, params=model.parameters()) # SGD and Adam both segfault

torch.manual_seed(0)

for i in range(1000):

    print(i)

    opt.zero_grad()

    x = Variable(torch.randn(64, 3, 32, 32))
    x = model(x)
    loss = x.sum()

    loss.backward() # segfault here

    opt.step()
