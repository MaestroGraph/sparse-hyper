import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import util, logging, time, gc
import numpy as np

import pickle

plt.figure(figsize=(10.5, 3.5))
plt.clf()

files  = [
    'results.4.np.npy',
    'results.8.np.npy',
    'results.16.np.npy'
]
sizes  = [4, 8, 16]
itss   = [30_000, 60_000, 120_000]
des    = [500, 500, 1000]

norm = mpl.colors.Normalize(vmin=2, vmax=6)
cmap = plt.get_cmap('Set1')
map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

labels = []
handles = []

for si, (file, size, iterations, dot_every) in enumerate(zip(files, sizes, itss, des)):
    res = np.load('./paper/sort/' + file)
    res = 1.0 - res  # acc to error
    print('size ', size, 'reinf', res.shape)

    color = map.to_rgba(np.log2(size))
    ndots = iterations // dot_every

    # print(size,  np.mean(res, axis=0))

    labels.append('{0}x{0}, r={1}'.format(size,  res.shape[0]))
    if res.shape[0] > 1:
        h = plt.errorbar(
            x=np.arange(ndots) * dot_every, y=np.mean(res, axis=0), yerr=np.std(res, axis=0),
            color=color)
        handles.append(h)

    else:
        h = plt.plot(
            np.arange(ndots) * dot_every, np.mean(res, axis=0),
            color=color)

        handles.append(h[0])

ax = plt.gca()
ax.set_ylim((0, 1))
ax.set_xlabel('iterations')
ax.set_ylabel('error')
ax.legend(handles, labels, loc='upper left', bbox_to_anchor= (0.96, 1.0), ncol=1,
            borderaxespad=0, frameon=False)

util.basic()
ax.spines["bottom"].set_visible(False)

plt.tight_layout()

plt.savefig('./paper/sort/sort.pdf', dpi=600)