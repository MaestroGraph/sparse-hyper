import hyper, gaussian, util, logging, time, pretrain
import torch, random
from torch.autograd import Variable
from torch import nn, optim
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from util import Lambda, Debug

from torchsample.metrics import CategoricalAccuracy

import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms

from util import od

from argparse import ArgumentParser

"""
MNIST experiment

"""

PLOT = True

def go(batch=64, epochs=350, k=750, additional=512, model_name='non-adaptive', cuda=False, seed=1,
       bias=True, data='./data', lr=0.01, lambd=0.01, subsample=None, deconvs=2, penalty=0.0):

    torch.manual_seed(seed)
    logging.basicConfig(filename='run.log',level=logging.INFO)
    LOG = logging.getLogger()
    plt.figure(figsize=(16, 4))

    w = SummaryWriter()

    SHAPE = (1, 28, 28)

    gaussian.PROPER_SAMPLING = False

    normalize = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=normalize)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=2)
    test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=normalize)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

    if model_name == 'non-adaptive':
        shapes = [SHAPE, (4, 16, 16), (8, 8, 8)]

        layer1   = nn.Sequential(gaussian.ParamASHLayer(shapes[0], shapes[1], k=k, additional=additional, has_bias=bias, subsample=subsample), nn.Sigmoid())
        decoder1 = nn.Sequential(gaussian.ParamASHLayer(shapes[1], shapes[0], k=k, additional=additional, has_bias=bias, subsample=subsample))

        layer2 = gaussian.ParamASHLayer(shapes[1], shapes[2], k=k, additional=additional, has_bias=bias, subsample=subsample)
        decoder2 = nn.Sequential(
            gaussian.ParamASHLayer(shapes[2], shapes[1], k=k, additional=additional, has_bias=bias, subsample=subsample),
            nn.Sigmoid())

        to_class = nn.Sequential(
            util.Flatten(),
            nn.Linear(512, 10),
            nn.Softmax())

        if cuda:
            layer1.apply(lambda t: t.cuda())
            layer2.apply(lambda t: t.cuda())

            decoder1.apply(lambda t: t.cuda())
            decoder2.apply(lambda t: t.cuda())

            to_class.cuda()

    elif model_name == 'free':
        shapes = [SHAPE, (4, 16, 16), (8, 8, 8)]

        layer1   = nn.Sequential(
            gaussian.CASHLayer(shapes[0], shapes[1], k=k, additional=additional, has_bias=bias, has_channels=True, subsample=subsample),
            nn.Sigmoid())

        decoder1 = nn.Sequential(gaussian.CASHLayer(shapes[1], shapes[0], k=k, additional=additional, has_bias=bias, has_channels=True, subsample=subsample))

        layer2 = gaussian.CASHLayer(shapes[1], shapes[2], k=k, additional=additional, has_bias=bias, has_channels=True, subsample=subsample)
        decoder2 = nn.Sequential(
            gaussian.CASHLayer(shapes[2], shapes[1], k=k,  additional=additional, has_bias=bias, has_channels=True, subsample=subsample),
            nn.Sigmoid())

        to_class = nn.Sequential(
            util.Flatten(),
            nn.Linear(512, 10),
            nn.Softmax())

        if cuda:
            layer1.apply(lambda t: t.cuda())
            layer2.apply(lambda t: t.cuda())

            decoder1.apply(lambda t: t.cuda())
            decoder2.apply(lambda t: t.cuda())

            to_class.cuda()

    elif model_name == 'free9':
        shapes = [SHAPE, (4, 16, 16), (8, 8, 8)]

        layer1   = nn.Sequential(
            gaussian.CASHLayer(shapes[0], shapes[1], k=k, ksize=9, deconvs=deconvs, additional=additional, has_bias=bias, has_channels=True, subsample=subsample),
            nn.Sigmoid())
        decoder1 = nn.Sequential(gaussian.CASHLayer(shapes[1], shapes[0], k=k,ksize=9, deconvs=deconvs, additional=additional, has_bias=bias, has_channels=True, subsample=subsample))

        layer2 = gaussian.CASHLayer(shapes[1], shapes[2], k=k, ksize=9, deconvs=deconvs, additional=additional, has_bias=bias, has_channels=True, subsample=subsample)
        decoder2 = nn.Sequential(
            gaussian.CASHLayer(shapes[2], shapes[1], k=k, ksize=9, deconvs=deconvs, additional=additional, has_bias=bias, has_channels=True, subsample=subsample),
            nn.Sigmoid())

        to_class = nn.Sequential(
            util.Flatten(),
            nn.Linear(512, 10),
            nn.Softmax())

        if cuda:
            layer1.apply(lambda t: t.cuda())
            layer2.apply(lambda t: t.cuda())

            decoder1.apply(lambda t: t.cuda())
            decoder2.apply(lambda t: t.cuda())

            to_class.cuda()

    else:
        raise Exception('Model {} not found'.format(model_name))

    ## SIMPLE
    xent = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    acc = CategoricalAccuracy()
    optimizer = optim.Adam(
        list(layer1.parameters())+ list(layer2.parameters()) + list(decoder1.parameters()) + list(decoder2.parameters()) +
        list(to_class.parameters()), lr=lr)

    step = 0

    for epoch in range(epochs):
        for i, data in tqdm(enumerate(trainloader, 0)):

            # get the inputs
            inputs, labels = data
            # inputs = inputs.squeeze(1)

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            # forward + backward + optimize
            optimizer.zero_grad()

            h1 = layer1(inputs)
            h2 = layer2(h1)
            outputs = to_class(h2)

            rec1 = decoder1(h1)
            rec2 = decoder1(decoder2(h2))

            cls_loss = xent(outputs, labels)
            rec_loss = mse(rec1, inputs.detach()) + mse(rec2, inputs.detach())
            sig_loss = layer1[0].sigma_loss(inputs) + layer2.sigma_loss(inputs)

            loss = cls_loss + lambd * rec_loss + penalty * sig_loss

            t0 = time.time()
            loss.backward()  # compute the gradients
            logging.info('backward: {} seconds'.format(time.time() - t0))
            optimizer.step()

            w.add_scalar('mnist/train-loss-cls', cls_loss.data[0], step)
            w.add_scalar('mnist/train-loss-rec', rec_loss.data[0], step)


            step += 1

            if PLOT and i == 0:

                rec1 = rec1.clamp(0.0, 1.0)
                rec2 = rec2.clamp(0.0, 1.0)

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(inputs.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('rec.{:03d}.input.pdf'.format(epoch))

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(rec1.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('rec.{:03d}.layer1.pdf'.format(epoch))

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(rec2.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('rec.{:03d}.layer2.pdf'.format(epoch))

        total = 0.0
        num = 0

        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            # inputs = inputs.squeeze(1)

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            # forward + backward + optimize
            optimizer.zero_grad()
            h1 = layer1(inputs)
            h2 = layer2(h1)
            outputs = to_class(h2)

            total += acc(outputs, labels)
            num += 1

        accuracy = total/num

        w.add_scalar('mnist/per-epoch-test-acc', accuracy, epoch)
        print('EPOCH {}: {} accuracy '.format(epoch, accuracy))

    print('Finished Training.')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Which model to train.",
                        default='non-adaptive')

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples",
                        default=750, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled",
                        default=512, type=int)

    parser.add_argument("-d", "--deconvs",
                        dest="deconvs",
                        help="Number of deconvolutions in adaptive model",
                        default=3, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.01, type=float)

    parser.add_argument("-L", "--lambda",
                        dest="lambd",
                        help="Reconstruction loss weight",
                        default=0.01, type=float)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-B", "--no-bias", dest="bias",
                        help="Whether to give the layers biases.",
                        action="store_false")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data')

    parser.add_argument("-S", "--subsample",
                        dest="subsample",
                        help="Sample a subset of the indices to estimate gradients for",
                        default=None, type=float)

    parser.add_argument("-P", "--penalty",
                        dest="penalty",
                        help="Penalty loss term multiplier",
                        default=0.0, type=float)

    options = parser.parse_args()

    print('OPTIONS', options)

    go(batch=options.batch_size, k=options.k, bias=options.bias, additional=options.additional,
       model_name=options.model, cuda=options.cuda, data=options.data, lr=options.lr,
       lambd=options.lambd, subsample=options.subsample, deconvs=options.deconvs, penalty=options.penalty)
