import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import util, logging, time, gc
import numpy as np

import pickle
dot_every = 100

plt.figure(figsize=(10, 5))
plt.clf()

files  = [
    'results.004.npy',
    'results.008.npy']
sizes  = [4, 8]
itss   = [8_000, 32_000]
des    = [500, 500]
reinfs = [False, False]

norm = mpl.colors.Normalize(vmin=min(np.log2(sizes)), vmax=max(np.log2(sizes)))
cmap = plt.get_cmap('viridis')
map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

for si, (file, size, iterations, dot_every, reinf) in enumerate(zip(files, sizes, itss, des, reinfs)):
    res = np.load('./paper/sort/' + file)
    print('size ', size, 'reinf', reinf, res.shape)

    color = map.to_rgba(np.log2(size))
    ndots = iterations // dot_every
    additional = int(np.floor(np.log2(size)) * size)

    # print(reinfo, res[rf, :, :])
    print(size, reinf, np.mean(res, axis=0))
    # print(reinforce, np.std(res[rf, :, :], axis=0))

    lbl = '{0}x{0}, a={1}, r={2}'.format(size, additional, res.shape[0])
    if res.shape[0] > 1:
        plt.errorbar(
            x=np.arange(ndots) * dot_every, y=np.mean(res, axis=0), yerr=np.std(res, axis=0),
            label=lbl, color=color)
    else:
        plt.plot(
            x=np.arange(ndots) * dot_every, y=np.mean(res, axis=0),
            label=lbl, color=color)

ax = plt.gca()
ax.set_ylim((0, 1))
ax.set_xlabel('error')
ax.set_ylabel('mean-squared error')
ax.legend()

util.basic()

plt.savefig('./paper/sort/sort.pdf', dpi=600)