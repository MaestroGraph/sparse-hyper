import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import util, logging, time, gc
import numpy as np

import pickle

"""
Used to create Figure 2 in the paper
"""

plt.figure(figsize=(10.5, 3.5))
plt.clf()

files  = [
    'results.004.True.npy',
    'results.008.True.npy',
    'results.016.True.npy',
    'results.008.False.npy',
    'results.016.False.npy',
    'results.032.False.npy',
    'results.064.False.npy',
]
sizes  = [4, 8, 16, 8, 16, 32, 64]
itss   = [120_000, 120_000, 120_000, 10_000, 20_000, 40_000, 120_000]
des    = [1000, 1000, 1000, 500, 500, 500, 500]
reinfs = [True, True, True, False, False, False, False]
lrs    = [0.005,0.005,0.005,0.005,0.005,0.005,0.001]

norm = mpl.colors.Normalize(vmin=2, vmax=6)
cmap = plt.get_cmap('Set1')
map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

handles, labels = [], []

for si, (file, size, iterations, dot_every, reinf, lr) in enumerate(zip(files, sizes, itss, des, reinfs, lrs)):
    res = np.load('./paper/identity/' + file)
    print('size ', size, 'reinf', reinf, res.shape)

    color = map.to_rgba(np.log2(size))
    ndots = iterations // dot_every

    print(ndots)

    lbl = '{0}x{0}, r={1}'.format(size, res.shape[0])
    if res.shape[0] > 1:
        h = plt.errorbar(
            x=np.arange(ndots) * dot_every, y=np.mean(res, axis=0), yerr=np.std(res, axis=0),
            label=lbl, color=color, linestyle='--' if reinf else '-',  alpha=0.2 if reinf else 1.0)
        handles.append(h)

    else:
        h = plt.plot(
            np.arange(ndots) * dot_every, np.mean(res, axis=0),
            label=lbl, color=color, linestyle='--' if reinf else '-')
        handles.append(h[0])

    labels.append(lbl)

ax = plt.gca()
ax.set_ylim(bottom=0)
ax.set_xlabel('iterations')
ax.set_ylabel('mean-squared error')
ax.legend(handles, labels, loc='upper left', bbox_to_anchor= (0.96, 1.0), ncol=1,
            borderaxespad=0, frameon=False)

util.basic()

ax.spines["bottom"].set_visible(False)

plt.tight_layout()

plt.savefig('./paper/identity/identity.pdf', dpi=600)