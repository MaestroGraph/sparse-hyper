import hyper, gaussian, util, logging, time, pretrain
import torch, random
from torch.autograd import Variable
from torch import nn, optim
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from util import Lambda, Debug

from torchsample.metrics import CategoricalAccuracy

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from util import od

from argparse import ArgumentParser

"""
MNIST experiment

"""

EPOCHS = 350
PLOT = True

def vae_loss(x, x_rec, mu, logvar):
    b, c, w, h = x.size()

    xent = nn.functional.binary_cross_entropy(x_rec.contiguous().view(-1, 784), x.contiguous().view(-1, 784))

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / (b * 784)


    return xent + kl


def go(batch=64, cuda=False, seed=1, lr=0.001, data='./data'):

    FT = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    torch.manual_seed(seed)
    logging.basicConfig(filename='run.log',level=logging.INFO)
    LOG = logging.getLogger()

    w = SummaryWriter()

    normalize = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=normalize)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=2)
    test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=normalize)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

    encoder = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=3),
        nn.ReLU(),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), # 1 by 1
        nn.ReLU(),
        util.Flatten(),
        nn.Linear(512, 64)
    )

    decoder = nn.Sequential(
        nn.Linear(32, 512), nn.ReLU(),
        util.Lambda(lambda x : x.unsqueeze(2).unsqueeze(3)),
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),  # 1 by 1
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),  # 2 by 2
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),  # 4 by 4
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),  # 8 by 8
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),  # 16 by 16
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=3),  # 32 by 32
        nn.Sigmoid(),
    )

    if cuda:
        encoder.cuda()
        decoder.cuda()

    ## SIMPLE
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    step = 0

    for epoch in range(EPOCHS):
        for i, data in tqdm(enumerate(trainloader, 0)):

            # get the inputs
            inputs, _ = data
            # inputs = inputs.squeeze(1)

            if cuda:
                inputs = inputs.cuda()

            # wrap them in Variables
            inputs = Variable(inputs)

            # forward + backward + optimize
            optimizer.zero_grad()

            z = encoder(inputs)
            mu = z[:, :32]
            logvar = z[:, 32:]

            sample = Variable(FT(mu.size()).normal_())

            std = logvar.mul(0.5).exp()
            sample = sample.mul(std).add(mu)

            reconstruction = decoder(sample)

            loss = vae_loss(inputs, reconstruction, mu, logvar)

            loss.backward()  # compute the gradients
            optimizer.step()

            w.add_scalar('mnist/train-loss', loss.data[0], step)

            step += 1
            if PLOT and i % 10 == 0:

                rec = reconstruction.clamp(0.0, 1.0)

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(inputs.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('vae.{:03d}.input.pdf'.format(epoch))

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(rec.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('vae.{:03d}.reconstruction.pdf'.format(epoch))

    print('Finished Training.')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)


    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data')

    options = parser.parse_args()

    print('OPTIONS', options)

    go(batch=options.batch_size, cuda=options.cuda, data=options.data, lr=options.lr )
