from time import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.axes import Axes

import torch
from torch import nn


import numpy as np

tics = []

def tic():
    tics.append(time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time()-tics.pop()

def plot(means, sigmas, values, axes=None):

    b, n, d = means.size()

    means = means.data[0, :,:].numpy()
    sigmas = sigmas.data[0, :].numpy()
    values = nn.functional.tanh(values).data[0, :].numpy()

    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    cmap = mpl.cm.RdYlBu
    map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    if axes is None:
        axes = plt.gca()

    colors = []
    for i in range(n):
        color = map.to_rgba(values[i])
        alpha = (sigmas[i]+1.0)**-2
        axes.add_patch(Circle((means[i, :]), radius=sigmas[i], color=color, alpha=alpha, linewidth=0))
        colors.append(color)

    axes.scatter(means[:, 0], means[:, 1], c=colors, zorder=100, linewidth=1, edgecolor='k')

def norm(x):
    """
    Normalize a tensor to a tensor with unit norm (treating first dim as batch dim)

    :param x:
    :return:
    """
    b = x.size()[0]

    n = torch.norm(x.view(b, -1), p=2, dim=1)
    while len(n.size()) < len(x.size()):
        n = n.unsqueeze(1)

    n.expand_as(x)

    return x/n