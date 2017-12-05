import hyper, gaussian, util, logging, time
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

from util import od

w = SummaryWriter()

def pretrain(layers, shapes, pivots, loader, epochs=50, plot=False, k_out=256, out_additional=128, out_has_bias=True, learn_rate=0.01, use_cuda=False, has_channels=True):

    ## SIMPLE
    criterion = nn.MSELoss()
    pre = []
    post = []

    for j, pivot in enumerate(pivots):

        print('pretraining, level ', j)

        mid_shape, out_shape = shapes[j+1], shapes[j]

        encoder = layers[:pivots[j]] if j == 0 else layers[pivots[j-1]:pivots[j]]

        decoder = [
            gaussian.CASHLayer(mid_shape, out_shape, k=k_out, additional=out_additional, has_bias=out_has_bias, has_channels=has_channels[j]),
            nn.Sigmoid()]

        model = nn.Sequential(od(encoder + decoder))
        pre_model = None if len(pre) == 0 else nn.Sequential(od(pre))
        post_model = None if len(post) == 0 else nn.Sequential(od(post))

        if use_cuda:
            model.apply(lambda t: t.cuda())

        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

        for epoch in range(epochs):
            for i, data in tqdm(enumerate(loader, 0)):

                # get the inputs
                inputs, _ = data
                inputs = inputs.squeeze(1)

                if use_cuda:
                    inputs = inputs.cuda()

                inputs = Variable(inputs)
                inputs_old = inputs

                if pre_model is not None:
                    inputs = pre_model(inputs)

                inputs.detach()
                targets = Variable(inputs.data)

                # forward + backward + optimize
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                t0 = time.time()
                loss.backward()  # compute the gradients
                logging.info('backward: {} seconds'.format(time.time() - t0))

                optimizer.step()

                w.add_scalar('pretrain/train-loss-{}'.format(j), loss.data[0], i)

                if plot and i == 0:

                    if post_model is not None:
                        outputs = post_model(outputs)

                    inputs, outputs = inputs_old.unsqueeze(1), outputs.unsqueeze(1)

                    plt.figure(figsize=(16, 4))
                    plt.imshow(np.transpose(torchvision.utils.make_grid(inputs.data[:16,:]).cpu().numpy(), (1, 2, 0)), interpolation='nearest')
                    plt.savefig('pretrain.input.{}.{:03d}.pdf'.format(j,epoch))

                    plt.figure(figsize=(16, 4))
                    plt.imshow(np.transpose(torchvision.utils.make_grid(outputs.data[:16,:]).cpu().numpy(), (1, 2, 0)), interpolation='nearest')
                    plt.savefig('pretrain.output.{}.{:03d}.pdf'.format(j,epoch))

            pre = pre + encoder
            post = decoder + post

        print('finished pretraining.')

