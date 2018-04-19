import hyper, gaussian, util, time, pretrain, os, math
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

def vae_loss(x, x_rec, mu, logvar):
    b, c, w, h = x.size()
    total = util.prod(x.size()[1:])

    xent = nn.functional.binary_cross_entropy(x_rec.contiguous().view(-1, total), x.contiguous().view(-1, total))

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / (b * total)

    return xent + kl

def generate_er(n=128, m=512, num=64):

    data = torch.FloatTensor(num, n, n)
    classes = torch.LongTensor(num)

    for i in range(num):
        graph = nx.gnm_random_graph(n, m)
        am = nx.to_numpy_matrix(graph)

        data[i, :, :] = torch.from_numpy(am)

    return data

SIZE = 60000
PLOT = True

def go(nodes=128, links=512, batch=64, epochs=350, k=750, additional=512, modelname='baseline', cuda=False, seed=1, bias=True, lr=0.001, lambd=0.01, subsample=None):

    FT = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    torch.manual_seed(seed)

    w = SummaryWriter()

    SHAPE = (1, nodes, nodes)

    LOG.info('generating data...')
    data = generate_er(nodes, links, SIZE)
    LOG.info('done.')

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

        zsize = 256
        shapes = [SHAPE, (4, 32, 32), (8, 4, 4), (1, zsize * 2), (1, zsize), (8, 4, 4), (4, 32, 32), SHAPE]

        layer = [None] * 6
        rec   = [None] * 5

        layer[0] = nn.Sequential(
            gaussian.CASHLayer(shapes[0], shapes[1], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None),
            nn.Sigmoid())
        rec[0] = nn.Sequential(
            gaussian.CASHLayer(shapes[1], shapes[0], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None))

        layer[1] = nn.Sequential(
            gaussian.CASHLayer(shapes[1], shapes[2], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None),
            nn.Sigmoid())
        rec[1] = nn.Sequential(
            gaussian.CASHLayer(shapes[2], shapes[1], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None),
            nn.Sigmoid())

        layer[2] = nn.Sequential(
            gaussian.CASHLayer(shapes[2], shapes[3], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None),
            nn.Sigmoid())
        rec[2] = nn.Sequential(
            gaussian.CASHLayer(shapes[3], shapes[2], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None),
            nn.Sigmoid())



        layer[3] = nn.Sequential(
            gaussian.CASHLayer(shapes[4], shapes[5], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None),
            nn.Sigmoid())
        rec[3] = nn.Sequential(
            gaussian.CASHLayer(shapes[5], shapes[4], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None),
            nn.Sigmoid())

        layer[4] = nn.Sequential(
            gaussian.CASHLayer(shapes[5], shapes[6], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None),
            nn.Sigmoid())
        rec[4] = nn.Sequential(
            gaussian.CASHLayer(shapes[6], shapes[5], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None),
            nn.Sigmoid())

        layer[5] = nn.Sequential(
            gaussian.CASHLayer(shapes[6], shapes[7], poolsize=1, k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False, subsample=None),
            nn.Sigmoid())

        if cuda:
            for l in layer:
                l.apply(lambda t: t.cuda())

            for r in rec:
                r.apply(lambda t: t.cuda())

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


    xent = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    acc = CategoricalAccuracy()


    parameters = []
    for l in layer:
        parameters.extend(l.parameters())
    for r in rec:
        parameters.extend(r.parameters())

    optimizer = optim.Adam(parameters, lr=lr)

    step = 0
    iterations = int(math.ceil(SIZE/batch))


    for epoch in range(epochs):
        for i in trange(iterations):

            # get the inputs
            f, t = i * batch, (i+1)*batch if i < iterations - 1 else SIZE
            graphs = data[f:t, :, :]

            if cuda:
                graphs = graphs.cuda()

            graphs = Variable(graphs.unsqueeze(1).contiguous())

            # forward + backward + optimize
            optimizer.zero_grad()

            h = [None] * 6

            h[0] = layer[0](graphs)
            h[1] = layer[1](h[0])
            h[2] = layer[2](h[1])

            mu, logvar = h[2][:, :, zsize:], h[2][:, :, zsize:]
            mu, logvar = mu.squeeze(1), logvar.squeeze(1)

            sample = Variable(FT(mu.size()).normal_())

            std = logvar.mul(0.5).exp()
            sample = sample.mul(std).add(mu)

            sample = sample.unsqueeze(1)
            sample_target = sample.detach()

            h[3] = layer[3](sample)
            h[4] = layer[4](h[3])
            reconstruction = layer[5](h[4])

            r = [None] * 6

            r[0] = rec[0](h[0])
            r[1] = rec[0](rec[1](h[1]))
            r[2] = rec[0](rec[1](rec[2](h[2])))

            r[3] = rec[3](h[3])
            r[4] = rec[3](rec[4](h[4]))

            rec_loss_encoder = mse(r[0], graphs) + mse(r[1], graphs) + mse(r[2], graphs)
            rec_loss_decoder = mse(r[3], sample_target) + mse(r[4], sample_target)

            vae = vae_loss(graphs, reconstruction, mu, logvar)

            loss = vae + lambd * (rec_loss_encoder + rec_loss_decoder)

            t0 = time.time()
            loss.backward()  # compute the gradients
            logging.info('backward: {} seconds'.format(time.time() - t0))
            optimizer.step()

            w.add_scalar('graphs/train-loss', loss.data[0], step)

            step += 1

            if PLOT and i == 0:


                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(graphs.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('gvae.{:03d}.input.pdf'.format(epoch))

                for i, recon in enumerate(r[:3]):

                    recon = recon.clamp(0.0, 1.0)

                    plt.cla()
                    plt.imshow(np.transpose(torchvision.utils.make_grid(recon.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                               interpolation='nearest')
                    plt.savefig('gvae.{:03d}.layer{:01d}.pdf'.format(epoch, i))

                reconstruction = reconstruction.clamp(0.0, 1.0)

                plt.cla()
                plt.imshow(np.transpose(torchvision.utils.make_grid(reconstruction.data[:16, :]).cpu().numpy(), (1, 2, 0)),
                           interpolation='nearest')
                plt.savefig('gvae.{:03d}.output.pdf'.format(epoch))

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

    parser.add_argument("-S", "--subsample",
                        dest="subsample",
                        help="Sample a subset of the indices to estimate gradients for",
                        default=None, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(batch=options.batch_size, nodes=options.nodes, links=options.links, k=options.k, bias=options.bias,
        additional=options.additional, modelname=options.model, cuda=options.cuda,
        lr=options.lr, lambd=options.lambd, subsample=options.subsample)
