import hyper, gaussian, util, logging, time, pretrain
import torch, random
from torch.autograd import Variable
from torch import nn, optim
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter

from torchsample.metrics import CategoricalAccuracy

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from util import od

torch.manual_seed(1)
logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()

"""
MNIST experiment

"""
w = SummaryWriter()

BATCH = 256
SHAPE = (28, 28)
EPOCHS = 350
PRETRAIN = True

CUDA = True

gaussian.PROPER_SAMPLING = False

TYPE = 'free'

normalize = transforms.Compose(
    [transforms.ToTensor()])

train = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=normalize)

trainloader = torch.utils.data.DataLoader(train, batch_size=BATCH,
                                          shuffle=True, num_workers=2)

test = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=normalize)

testloader = torch.utils.data.DataLoader(test, batch_size=BATCH,
                                         shuffle=False, num_workers=2)

if TYPE == 'non-adaptive':
    layers = [
        gaussian.ParamASHLayer(SHAPE, (4, 8, 8), k=64, additional=8, has_bias=True),
        nn.Sigmoid(),
        gaussian.ParamASHLayer((4, 8, 8), (128,), k=64, additional=32, has_bias=True),
        nn.Sigmoid(),
        nn.Linear(128, 10),
        nn.Softmax()]
elif TYPE == 'free':

    shapes = [(28, 28), (4, 8, 8), (8, 4, 4), (128,), (10,)]
    layers = [
        gaussian.CASHLayer(shapes[0], shapes[1], k=256, additional=64, has_bias=True),
        nn.Sigmoid(),
        gaussian.CASHLayer(shapes[1], shapes[2], k=128, additional=32, has_bias=True),
        nn.Sigmoid(),
        gaussian.CASHLayer(shapes[2], shapes[3], k=128, additional=32, has_bias=True),
        nn.Sigmoid(),
        nn.Linear(shapes[3][0], shapes[4][0]),
        nn.Softmax()]
    pivots = [2, 4, 6, 7]

if PRETRAIN:
    pretrain.pretrain(layers, shapes, pivots, trainloader, epochs=20, k_out=256, out_additional=64, use_cuda=CUDA, plot=True)


model = nn.Sequential(od(layers))

if CUDA:
    model.apply(lambda t : t.cuda())

## SIMPLE
criterion = nn.CrossEntropyLoss()
acc = CategoricalAccuracy()
optimizer = optim.Adam(model.parameters(), lr=0.01)

step = 0

for epoch in range(EPOCHS):
    for i, data in tqdm(enumerate(trainloader, 0)):

        # get the inputs
        inputs, labels = data
        inputs = inputs.squeeze(1)

        if CUDA:
            inputs, labels = inputs.cuda(), labels.cuda()

        # wrap them in Variables
        inputs, labels = Variable(inputs), Variable(labels)

        # forward + backward + optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        t0 = time.time()
        loss.backward()  # compute the gradients
        logging.info('backward: {} seconds'.format(time.time() - t0))
        optimizer.step()

        w.add_scalar('mnist/train-loss', loss.data[0], step)

        step += 1

    total = 0.0
    num = 0
    for i, data in enumerate(testloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.squeeze(1)

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

