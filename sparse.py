import torch
from numpy.core.multiarray import dtype
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import torchsample as ts
from torchsample.modules import ModuleTrainer

from torchsample.metrics import *

import time

EPOCHS = 350
BATCH_SIZE = 64
GPU = False

SMALL = 96
BIG = 192

# Set up the dataset

normalize = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=normalize)

trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=normalize)

testloader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class HyperLayer(nn.Module):

    def __init__(self, in_size : int, out_size : int, hyper_size = 16):
        super(HyperLayer, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.hyper = torch.nn.Sequential(
            torch.nn.Linear(3, hyper_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hyper_size, 1),
        )

        self.hyper_input = Variable(torch.ones( (in_size * out_size, 3)), requires_grad=False)

        index = 0
        for i in range(self.in_size):
            for j in range(self.out_size):
                self.hyper_input[index, 0] = i
                self.hyper_input[index, 1] = j
                self.hyper_input[index, 2] = 1 # bias node

                index += 1

    def forward(self, x):

        w_flat = self.hyper(self.hyper_input)
        self.w = w_flat.view(self.in_size, self.out_size)

        return x.mm(self.w)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.hyper1 = HyperLayer(in_size = 3072, out_size=1024)
        self.hyper2 = HyperLayer(in_size = 1024, out_size=512)
        self.hyper3 = HyperLayer(in_size = 512, out_size=256)
        #self.hyper4 = HyperLayer(in_size = 128, out_size=256)

        self.fin = nn.Linear(256, 10)

    def forward(self, x):

        x = x.view(-1, self.num_flat_features(x)) # flatten

        x = self.hyper1(x)
        x = F.relu(x)
        x = self.hyper2(x)
        x = F.relu(x)
        x = self.hyper3(x)
        x = F.relu(x)
        # x = self.hyper4(x)
        # x = F.relu(x)

        x = self.fin(x)
        x = F.softmax(x)

        # print(self.hyper4.w[1:5, 1:5])
        # print(list(self.hyper4.hyper[0].parameters())[:3])

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model = Net()

if GPU:
    model = model.cuda(0)

# print(model)

trainer = ModuleTrainer(model)

trainer.compile(
    loss=nn.CrossEntropyLoss(),
    # optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001),
    optimizer='adam',
    metrics=[CategoricalAccuracy()])

trainer.fit_loader(trainloader, testloader,
            nb_epoch=EPOCHS,
            verbose=1,
            cuda_device= 0 if GPU else -1)