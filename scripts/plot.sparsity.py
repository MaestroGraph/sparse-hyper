from _context import sparse
from sparse import util

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import logging, time, gc
import numpy as np

import pickle

"""
Plot the results of the sparsity experiment.
"""
plt.figure(figsize=(10.5, 3.5))
plt.clf()

models = ['nas-temp', 'l5', 'l2', 'l1']
name = {'nas-temp':'sparse layer', 'l1':'$l^1$', 'l2':'$l^\\frac{1}{2}$', 'l5':'$l^\\frac{1}{5}$'}
controls = 5

norm = mpl.colors.Normalize(vmin=0, vmax=5)
cmap = plt.get_cmap('Set1')
map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

handles, labels = [], []

for mi, model in enumerate(models):


    densities  = []
    accuracies = []

    dstds = []
    astds = []

    for c in range(controls):
        try:
            res = np.genfromtxt('results.{}.{}.csv'.format(model, c))
            if len(res.shape) == 1:
                res = res[None, :]

            assert len(res.shape) == 2 and res.shape[1] == 2

            repeats = res.shape[0]

            if repeats == 1:
                accuracies.append( res[0, 0])
                densities.append(res[0, 1])
            else:
                accuracies.append( res[:, 0].mean())
                densities.append( res[:, 1].mean())

                astds.append( res[:, 0].std())
                dstds.append( res[:, 1].std())
        except:
            print('could not load file results.{}.{}.csv'.format(model, c))

    color = map.to_rgba(mi)

    lbl = '{}, r={}'.format(name[model], repeats)
    labels.append(lbl)

    if len(dstds) == 0:
        h = plt.plot(
            densities, accuracies,
            label=lbl, linestyle='-' if model == 'nas-temp' else ':', marker='s')
        handles.append(h[0])
    else:
        h = plt.errorbar(
            x=densities, y=accuracies, xerr=dstds, yerr=astds,
            label=lbl, linestyle='-' if model == 'nas-temp' else ':')
        handles.append(h)



ax = plt.gca()

ax.set_xlabel('density')
ax.set_xscale('log')

ax.set_ylim(0, 1)
ax.set_ylabel('accuracy')

ax.legend(handles, labels, loc='upper left', bbox_to_anchor= (0.96, 1.0), ncol=1,
            borderaxespad=0, frameon=False)

util.basic()

plt.tight_layout()

plt.savefig('sparsity.pdf', dpi=600)