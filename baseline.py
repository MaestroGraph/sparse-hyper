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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=SMALL, kernel_size=3, padding=1) # out res 32x32
        self.conv2 = nn.Conv2d(in_channels=SMALL, out_channels=SMALL, kernel_size=3, padding=1) # out res 32x32
        self.conv3 = nn.Conv2d(in_channels=SMALL, out_channels=SMALL, kernel_size=3, stride=2, padding=1) # out res 16x16
        self.conv4 = nn.Conv2d(in_channels=SMALL, out_channels=BIG, kernel_size=3, padding=1) # out res 16
        self.conv5 = nn.Conv2d(in_channels=BIG, out_channels=BIG, kernel_size=3, padding=1) # out res 16
        self.conv6 = nn.Conv2d(in_channels=BIG, out_channels=BIG, kernel_size=3, stride=2, padding=1) # out res 8x8
        self.conv7 = nn.Conv2d(in_channels=BIG, out_channels=BIG, kernel_size=3, padding=1) # out res 8
        self.conv8 = nn.Conv2d(in_channels=BIG, out_channels=BIG, kernel_size=1, padding=1) # out res 8
        self.conv9 = nn.Conv2d(in_channels=BIG, out_channels=128, kernel_size=1, padding=1) # out res 8

        self.fin = nn.Linear(128, 10)

    def forward(self, x):

        # Max pooling over a (2, 2) window
        x = nn.Dropout2d(0.2)(x)
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = nn.Dropout2d(0.5)(F.relu(self.conv3(x)))
        x = (F.relu(self.conv4(x)))
        x = (F.relu(self.conv5(x)))
        x = nn.Dropout2d(0.5)(F.relu(self.conv6(x)))
        x = (F.relu(self.conv7(x)))
        x = (F.relu(self.conv8(x)))
        x = (F.relu(self.conv9(x)))

        x = F.max_pool2d(x, kernel_size=x.size()[2:])

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
#    optimizer='adam',
    optimizer=optim.SGD(model.fin.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001),
    metrics=[CategoricalAccuracy()])

trainer.fit_loader(trainloader, testloader,
            nb_epoch=EPOCHS,
            verbose=1,
            cuda_device= 0 if GPU else -1)

# for epoch in range(EPOCHS):  # loop over the dataset multiple times
#
#     running_loss = 0.0
#     batches = 0
#
#     start_time = time.time()
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs
#         inputs, labels = data
#
#         if GPU:
#             inputs = inputs.cuda()
#             labels = labels.cuda()
#
#         # wrap them in Variable
#         inputs, labels = Variable(inputs), Variable(labels)
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.data[0]
#         batches += 1
#     elapsed = time.time() - start_time
#     print('epoch %d, loss: %.3f, time taken %.3f seconds' % (epoch + 1, running_loss / batches, elapsed))
#
# print('Finished Training')