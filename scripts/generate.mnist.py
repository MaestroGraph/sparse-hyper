import torch
import torchvision
from torchvision.transforms import ToTensor

from _context import sparse
from sparse import util

from tqdm import tqdm
import numpy as np

import random, sys

from PIL import Image

from argparse import ArgumentParser

from math import ceil

from collections import Counter

"""
Generate rotated, and scaled version of MNIST
"""

def paste(background, foreground, scale=[1.5, 2.0]):

    rh, rw = background.size

    sch, scw = scale
    new_size = (int(ceil(foreground.size[0] * sch)), int(ceil(foreground.size[1] * scw)))
    try:
        foreground = foreground.resize(new_size, resample=Image.BICUBIC)
    except:
        print(f'pasting with size {new_size} failed. ({(sch, scw)})')
        sys.exit()

    # Rotate the foreground
    angle_degrees = random.randint(0, 359)
    foreground = foreground.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)

    h, w = foreground.size
    h, w = rh - h, rw - w
    h, w = random.randint(0, h), random.randint(0, w)

    background.paste(foreground, box=(h, w), mask=foreground)

def make_image(b, images, res=100, noise=10, scale=[1.0, 2.0], aspect=2.0):
    """

    Extract the b-th image from the batch of images, and place it into a 100x100 image, rotated and scaled
    with noise extracted from other images.

    :param b:
    :param images:
    :param res:
    :return:
    """


    # Sample a uniform scale for the noise and target
    sr = scale[1] - scale[0]
    ar = aspect - 1.0
    sc = random.random() * sr + scale[0]
    ap = random.random() * ar + 1.0

    sch, scw = sc, sc
    if random.choice([True, False]):
        sch *= ap
    else:
        scw *= ap

    background = Image.new(mode='RGB', size=(res, res))

    # generate random patch size
    nm = 14
    nh, nw = random.randint(4, nm), random.randint(4, nm)

    # Paste noise
    for i in range(noise):

        # select another image
        ind = random.randint(0, images.size(0)-2)
        if ind == b:
            ind += 1

        # clip out a random nh x nw patch
        h, w = random.randint(0, 28-nh), random.randint(0, 28-nw)
        nump = (images[ind, 0, h:h+nh, h:h+nw].numpy() * 255).astype('uint8').squeeze()
        patch = Image.fromarray(nump)

        paste(background, patch, scale=(sch, scw))

    # Paste image

    nump = (images[b, 0, :, :].numpy() * 255).astype('uint8').squeeze()

    foreground = Image.fromarray(nump)

    paste(background, foreground, scale=(sch, scw))

    return background

def go(arg):

    # make directories
    for i in range(10):
        util.makedirs('./mnist-rsc/train/{}/'.format(i))
        util.makedirs('./mnist-rsc/test/{}/'.format(i))

    train = torchvision.datasets.MNIST(root=arg.data, train=True, download=True, transform=ToTensor())
    trainloader = torch.utils.data.DataLoader(train, batch_size=arg.batch, shuffle=True, num_workers=2)

    test = torchvision.datasets.MNIST(root=arg.data, train=False, download=True, transform=ToTensor())
    testloader = torch.utils.data.DataLoader(test, batch_size=arg.batch, shuffle=True, num_workers=2)

    indices = Counter()

    for images, labels in tqdm(trainloader):

        batch_size = labels.size(0)

        for b in range(batch_size):
            image = make_image(b, images, res=arg.res, noise=arg.noise, scale=arg.scale)
            label = int(labels[b].item())

            image.save('./mnist-rsc/train/{}/{:06}.png'.format(label, indices[label]))

            indices[label] += 1

    indices = Counter()

    for images, labels in tqdm(testloader):

        batch_size = labels.size(0)

        for b in range(batch_size):
            image = make_image(b, images, res=arg.res, noise=arg.noise, scale=arg.scale)
            label = int(labels[b].item())

            image.save('./mnist-rsc/test/{}/{:06}.png'.format(label, indices[label]))

            indices[label] += 1

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data/')

    parser.add_argument("-b", "--batch-size",
                        dest="batch",
                        help="The batch size.",
                        default=256, type=int)

    parser.add_argument("-r", "--resolution",
                        dest="res",
                        help="Resolution (one side, images are always square).",
                        default=100, type=int)

    parser.add_argument("-n", "--noise",
                        dest="noise",
                        help="Number of noise patches to add.",
                        default=10, type=int)


    parser.add_argument("-s", "--scale",
                        dest="scale",
                        help="Min/max scale multiplier.",
                        nargs=2,
                        default=[1.0, 2.0], type=float)

    parser.add_argument("-a", "--aspect",
                        dest="aspect",
                        help="Min/max aspect multiplier.",
                        default=2.0, type=float)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)