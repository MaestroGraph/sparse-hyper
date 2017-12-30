import hyper, gaussian, util, time, pretrain, os
import torch, random
from torch.autograd import Variable
from torch import nn, optim
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from util import Lambda, Debug

from torch.utils.data import TensorDataset, DataLoader

from torchsample.metrics import CategoricalAccuracy

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from util import od, prod

from argparse import ArgumentParser

import networkx as nx

import logging

import matplotlib.pyplot as plt
import numpy as np

LOG = logging.getLogger('ash')
LOG.setLevel(logging.INFO)
fh = logging.FileHandler('ash.log')
fh.setLevel(logging.INFO)
LOG.addHandler(fh)

"""
Graph isomorphism experiment

"""

def generate(n=128, m=512, num=64):

    data = torch.FloatTensor(num, 2, n, n)
    classes = torch.LongTensor(num)

    for i in range(num):
        graph1 = nx.gnm_random_graph(n, m)
        am1 = nx.to_numpy_matrix(graph1)

        if random.choice([True, False]):
            # graphs are isomorphic
            nodes = list(graph1.nodes())
            random.shuffle(nodes)

            am2 = nx.to_numpy_matrix(graph1, nodelist=nodes)

            classes[i] = 0
        else:
            # graphs are (probably not) isomorphic
            graph2 = nx.gnm_random_graph(n, m)
            am2 = nx.to_numpy_matrix(graph2)

            classes[i] = 1

        data[i, 0, :, :] = torch.from_numpy(am1)
        data[i, 1, :, :] = torch.from_numpy(am2)

    return data, classes

TRAIN_SIZE = 60000
TEST_SIZE = 15000
PLOT = True

def go(nodes=128, links=512, batch=64, epochs=350, k=750, additional=512, modelname='baseline', cuda=False, seed=1, bias=True, lr=0.001, lambd=0.01):

    torch.manual_seed(seed)

    w = SummaryWriter()

    SHAPE = (1, nodes, nodes)

    LOG.info('generating data...')
    train, train_labels = generate(nodes, links, TRAIN_SIZE)
    test, test_labels = generate(nodes, links, TRAIN_SIZE)
    LOG.info('done.')

    ds_pretrain = TensorDataset(train.view(-1, 1, nodes, nodes), torch.zeros(train.size()[0] * 2))
    ds_train = TensorDataset(train, train_labels)
    ds_test = TensorDataset(test, test_labels)

    pretrain_loader = DataLoader(ds_pretrain, batch_size=batch, shuffle=True)
    train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch, shuffle=True)

    if modelname == 'non-adaptive':
    #     shapes = [SHAPE, (4, 32, 32), (8, 4, 4)]
    #     layers = [
    #         gaussian.ParamASHLayer(shapes[0], shapes[1], k=k, additional=additional, has_bias=bias),
    #         nn.Sigmoid(),
    #         gaussian.ParamASHLayer(shapes[1], shapes[2], k=k, additional=additional, has_bias=bias),
    #         nn.Sigmoid(),
    #         util.Flatten(),
    #         nn.Linear(128, 32),
    #         nn.Sigmoid()]
    #     pivots = [2, 4]
    #     decoder_channels = [True, True]
    #
    #     pretrain.pretrain(layers, shapes, pivots, pretrain_loader, epochs=pretrain_epochs, k_out=k, out_additional=additional, use_cuda=cuda,
    #             plot=True, out_has_bias=bias, has_channels=decoder_channels, learn_rate=pretrain_lr)
    #
    #     model = nn.Sequential(od(layers))
    #
    #     if cuda:
    #         model.apply(lambda t: t.cuda())
        pass

    elif modelname == 'free':

        shapes = [SHAPE, (4, 32, 32), (8, 4, 4)]

        layer1 = nn.Sequential(gaussian.CASHLayer(shapes[0], shapes[1], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False), nn.Sigmoid())
        decoder1 = nn.Sequential(gaussian.CASHLayer(shapes[1], shapes[0], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False))


        layer2 = nn.Sequential(gaussian.CASHLayer(shapes[1], shapes[2], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False), nn.Sigmoid())
        decoder2 = nn.Sequential(gaussian.CASHLayer(shapes[2], shapes[1], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False), nn.Sigmoid())

        to_class = nn.Sequential(
            util.Flatten(),
            nn.Linear(128, 32),
            nn.Sigmoid())

        if cuda:
            layer1.apply(lambda t: t.cuda())
            layer2.apply(lambda t: t.cuda())

            decoder1.apply(lambda t: t.cuda())
            decoder2.apply(lambda t: t.cuda())

            to_class.cuda()

    elif modelname == 'baseline':
        # model = nn.Sequential(
        #     # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
        #     nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=1, padding=2),
        #     nn.MaxPool2d(stride=4, kernel_size=4),
        #     # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))), # (4, 32, 32)
        #     nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=1, padding=2),
        #     nn.MaxPool2d(stride=4, kernel_size=4),
        #     # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))), # (8, 8, 8)
        #     # nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=1, padding=2),
        #     # nn.MaxPool2d(stride=4, kernel_size=4),
        #     # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
        #     util.Flatten(),
        #     nn.Linear(512, 32),
        #     nn.Sigmoid())
        #
        # if cuda:
        #     model = model.cuda()
        pass

    elif modelname == 'baseline-big':
        # model = nn.Sequential(
        #     # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
        #     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=2),
        #     nn.MaxPool2d(stride=2, kernel_size=2),
        #     # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))), # (4, 32, 32)
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=2),
        #     nn.MaxPool2d(stride=2, kernel_size=2),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=2),
        #     nn.MaxPool2d(stride=2, kernel_size=2),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
        #     nn.MaxPool2d(stride=2, kernel_size=2),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
        #     nn.MaxPool2d(stride=2, kernel_size=2),
        #     #Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))), # (64, 8, 8)
        #
        #     util.Flatten(),
        #     nn.Linear(1024, 32),
        #     nn.Sigmoid())
        #
        # if cuda:
        #     model = model.cuda()
        pass

    cls_decoder = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.Softmax()
    )

    xent = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    acc = CategoricalAccuracy()
    optimizer = optim.Adam(
        list(layer1.parameters())+ list(layer2.parameters()) + list(decoder1.parameters()) + list(decoder2.parameters()) +
        list(to_class.parameters()), lr=lr)

    step = 0

    for epoch in range(epochs):
        for i, data in tqdm(enumerate(train_loader, 0)):

            # get the inputs
            graphs, labels = data
            if cuda:
                graphs, labels = graphs.cuda(), labels.cuda()
            graphs1, graphs2 = graphs[:, 0, :], graphs[:, 1, :]

            graphs1, graphs2 = graphs1.unsqueeze(1), graphs2.unsqueeze(1)
            graphs1, graphs2, labels = Variable(graphs1), Variable(graphs2), Variable(labels)

            # forward + backward + optimize
            optimizer.zero_grad()

            h11 = layer1(graphs1.contiguous())
            h12 = layer2(h11)
            rep1 = to_class(h12)

            rec11 = decoder1(h11)
            rec12 = decoder1(decoder2(h12))

            h21 = layer1(graphs2.contiguous())
            h22 = layer2(h21)
            rep2 = to_class(h22)

            rec21 = decoder1(h21)
            rec22 = decoder1(decoder2(h22))

            rep = torch.cat((rep1, rep2), dim=1)
            outputs = cls_decoder(rep)

            cls_loss = xent(outputs, labels)
            rec_loss = \
                mse(rec11, graphs1.detach()) + mse(rec12, graphs1.detach()) + \
                mse(rec21, graphs2.detach()) + mse(rec22, graphs2.detach())

            loss = cls_loss + lambd * rec_loss

            t0 = time.time()
            loss.backward()  # compute the gradients
            logging.info('backward: {} seconds'.format(time.time() - t0))
            optimizer.step()

            w.add_scalar('graphs/train-loss', loss.data[0], step)

            step += 1

            if PLOT and i == 0:

                rec11 = rec11.clamp(0.0, 1.0)
                rec12 = rec12.clamp(0.0, 1.0)

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(graphs1.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('iso.{:03d}.input.pdf'.format(epoch))

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(rec11.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('iso.{:03d}.layer1.pdf'.format(epoch))

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(rec12.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('iso.{:03d}.layer2.pdf'.format(epoch))

        total = 0.0
        num = 0
        for i, data in tqdm(enumerate(test_loader, 0)):

            # get the inputs
            graphs, labels = data
            if cuda:
                graphs, labels = graphs.cuda(), labels.cuda()
            graphs1, graphs2 = graphs[:, 0, :], graphs[:, 1, :]

            graphs1, graphs2 = graphs1.unsqueeze(1), graphs2.unsqueeze(1)
            graphs1, graphs2, labels = Variable(graphs1), Variable(graphs2), Variable(labels)

            # forward + backward + optimize
            optimizer.zero_grad()
            h11 = layer1(graphs1.contiguous())
            h12 = layer2(h11)
            rep1 = to_class(h12)

            rec11 = decoder1(h11)
            rec12 = decoder1(decoder2(h12))

            h21 = layer1(graphs2.contiguous())
            h22 = layer2(h21)
            rep2 = to_class(h22)

            rec21 = decoder1(h21)
            rec22 = decoder1(decoder2(h22))

            rep = torch.cat((rep1, rep2), dim=1)
            outputs = cls_decoder(rep)

            total += acc(outputs, labels)
            num += 1

        accuracy = total/num

        w.add_scalar('graphs/per-epoch-test-acc', accuracy, epoch)
        LOG.info('EPOCH {}: {} accuracy '.format(epoch, accuracy))

    LOG.info('Finished Training.')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-N", "--nodes",
                        dest="nodes",
                        help="Number of nodes in the generated graphs.",
                        default=128, type=int)

    parser.add_argument("-M", "--links",
                        dest="links",
                        help="Number of links in the generated graphs.",
                        default=256, type=int)

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Which model to train.",
                        default='baseline')

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

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-B", "--no-bias", dest="bias",
                        help="Whether to give the layers biases.",
                        action="store_false")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data')

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.001, type=float)

    parser.add_argument("-L", "--lambda",
                        dest="lambd",
                        help="Reconstruction loss weight",
                        default=0.01, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(batch=options.batch_size, nodes=options.nodes, links=options.links, k=options.k, bias=options.bias,
        additional=options.additional, modelname=options.model, cuda=options.cuda,
        lr=options.lr, lambd=options.lambd)
