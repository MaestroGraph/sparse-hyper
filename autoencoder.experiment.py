import hyper, gaussian, util
import torch, random
from torch.autograd import Variable
from torch import nn, optim
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter

from torchsample.metrics import CategoricalAccuracy

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

"""
MNIST Autoencoder Experiment

"""
w = SummaryWriter()

BATCH = 256
SHAPE = (1, 28, 28)
MIDDLE = (8, )
EPOCHS = 350

CUDA = True

TYPE = 'free-weights'

normalize = transforms.Compose(
    [transforms.ToTensor()
        #,transforms.Normalize((0.1307,), (0.3081,))
     ])

train = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=normalize)

trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH,
                                          shuffle=True, num_workers=2)

test = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=normalize)

testloader = torch.utils.data.DataLoader(test, batch_size=BATCH,
                                         shuffle=False, num_workers=2)

if TYPE == 'non-adaptive':
    model = nn.Sequential(
        gaussian.ParamASHLayer(SHAPE, MIDDLE, k=16, additional=8, has_bias=True),
        nn.ReLU(),
        gaussian.ParamASHLayer(MIDDLE, SHAPE, k=16, additional=8, has_bias=True),
        nn.Sigmoid())
elif TYPE == 'free-weights':
    model = nn.Sequential(
        gaussian.CASHLayer(SHAPE, MIDDLE, k=16, additional=8, has_bias=True),
        nn.ReLU(),
        gaussian.CASHLayer(MIDDLE, SHAPE, k=16, additional=8, has_bias=True),
        nn.Sigmoid())

if CUDA:
    model.apply(lambda t : t.cuda())

## SIMPLE
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

step = 0

for epoch in range(EPOCHS):
    for i, data in tqdm(enumerate(trainloader, 0)):

        # get the inputs
        inputs, _ = data
        if CUDA:
            inputs = inputs.cuda()

        # wrap them in Variables
        inputs, targets = Variable(inputs), Variable(inputs)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        w.add_scalar('autoencoder/train-loss', loss.data[0], step)

        step += 1

    total = 0.0
    num = 0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, _ = data
        if CUDA:
            inputs = inputs.cuda()

        # wrap them in Variables
        inputs, targets = Variable(inputs), Variable(inputs)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total += loss

        num += 1

        if i < 10:

            plt.figure(figsize=(16, 4))
            plt.imshow(np.transpose(torchvision.utils.make_grid(inputs.data[:16,:]).cpu().numpy(), (1, 2, 0)), interpolation='nearest')
            plt.savefig('input.{}.{}.pdf'.format(epoch, i))

            plt.figure(figsize=(16, 4))
            plt.imshow(np.transpose(torchvision.utils.make_grid(outputs.data[:16,:]).cpu().numpy(), (1, 2, 0)), interpolation='nearest')
            plt.savefig('output.{}.{}.pdf'.format(epoch, i))

    epoch_loss = total/(num * BATCH)

    w.add_scalar('autoencoder/epoch-test-loss', epoch_loss.data[0], epoch)
    print('EPOCH {}: {} loss per instance '.format(epoch, epoch_loss.data[0]))

print('Finished Training.')

