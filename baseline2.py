import torch
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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2) # out res 32x32
        self.pool1 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5, stride=1, padding=2) # out res 16x16
        self.pool2 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2) # out res 16x16
        self.pool3 = nn.MaxPool2d(stride=2, kernel_size=2)

        self.fin = nn.Linear(4 * 4 * 20, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(-1, self.num_flat_features(x)) # flatten

        x = self.fin(x)

        x = F.softmax(x)

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

print(model)

trainer = ModuleTrainer(model)

trainer.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer='adam',
#    optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001),
    metrics=[CategoricalAccuracy()])

trainer.fit_loader(trainloader, testloader,
            num_epoch=EPOCHS,
            verbose=1,
            cuda_device= 0 if GPU else -1)

