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

torch.manual_seed(1)

"""
MNIST experiment

"""
w = SummaryWriter()

BATCH = 4
SHAPE = (1, 28, 28)
EPOCHS = 350

CUDA = False

TYPE = 'non-adaptive'

normalize = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
        gaussian.ParamASHLayer(SHAPE, (4, 8, 8), k=512, additional=512, has_bias=True),
        nn.ReLU(),
        gaussian.ParamASHLayer((4, 8, 8), (1024,), k=512, additional=512, has_bias=True),
        nn.ReLU(),
        gaussian.ParamASHLayer((1024,), (10,), k=512, additional=512, has_bias=True),
        nn.Softmax())

if CUDA:
    model.apply(lambda t : t.cuda())

## SIMPLE
criterion = nn.CrossEntropyLoss()
acc = CategoricalAccuracy()
optimizer = optim.Adam(model.parameters(), lr=0.001)

step = 0

for epoch in range(EPOCHS):
    for i, data in tqdm(enumerate(trainloader, 0)):

        # get the inputs
        inputs, labels = data
        if CUDA:
            inputs, labels = inputs.cuda(), labels.cuda()

        # wrap them in Variables
        inputs, labels = Variable(inputs), Variable(labels)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        w.add_scalar('mnist/train-loss', loss.data[0], step)

        step += 1

    total = 0.0
    num = 0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        if CUDA:
            inputs, labels = inputs.cuda(), labels.cuda()

        # wrap them in Variables
        inputs, labels = Variable(inputs), Variable(labels)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)

        total += acc(outputs, labels)
        num += 1
    accuracy = total/num

    w.add_scalar('mnist/per-epoch-test-acc', accuracy, epoch)
    print('EPOCH {}: {} accuracy '.format(epoch, accuracy))

print('Finished Training.')

