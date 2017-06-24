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

EPOCHS = 10
BATCH_SIZE = 256

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
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=2, padding=2) # out res 16x16

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=2, padding=1) # out res 8x8

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1) # out res 4x4

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1) # out res 2x2

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1) # out res 1x1

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1) # out res 1x1

        self.linear = nn.Linear(128, 10)  # out res 1x1

    def forward(self, x):

        # Max pooling over a (2, 2) window
        x = nn.Dropout2d(0.2)(F.relu(self.conv1(x)))
        x = nn.Dropout2d(0.2)(F.relu(self.conv2(x)))
        x = nn.Dropout2d(0.2)(F.relu(self.conv3(x)))
        x = nn.Dropout2d(0.2)(F.relu(self.conv4(x)))
        x = nn.Dropout2d(0.2)(F.relu(self.conv5(x)))
        x = nn.Dropout2d(0.2)(F.relu(self.conv6(x)))

        x = x.view(-1, self.num_flat_features(x)) # flatten

        x = self.linear(x)

        x = F.softmax(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



model = Net()

print(model)

trainer = ModuleTrainer(model)

trainer.compile(loss=nn.CrossEntropyLoss(), optimizer='adadelta', metrics=[CategoricalAccuracy()])

trainer.fit_loader(trainloader, testloader,
            nb_epoch=EPOCHS,
            verbose=1)

