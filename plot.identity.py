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
    'results.004.True.npy',
    'results.008.False.npy',
    'results.016.False.npy']
sizes  = [4, 8, 16]
itss   = [16_000, 32_000, 64_000]
des    = [500, 500, 500]
reinfs = [True, False, False]

norm = mpl.colors.Normalize(vmin=min(np.log2(sizes)), vmax=max(np.log2(sizes)))
cmap = plt.get_cmap('viridis')
map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

for si, (file, size, iterations, dot_every, reinf) in enumerate(zip(files, sizes, itss, des, reinfs)):
    res = np.load('./paper/identity/' + file)
    print('size ', size, 'reinf', reinf, res.shape)

    color = map.to_rgba(np.log2(size))
    ndots = iterations // dot_every
    additional = int(np.floor(np.log2(size)) * size)

    # print(reinfo, res[rf, :, :])
    print(size, reinf, np.mean(res, axis=0))
    # print(reinforce, np.std(res[rf, :, :], axis=0))

    lbl = '{0}x{0}, a={1}, r={2}, {3}'.format(size, additional, res.shape[0], 'reinforce' if reinf else 'backprop')
    if res.shape[0] > 1:
        plt.errorbar(
            x=np.arange(ndots) * dot_every, y=np.mean(res, axis=0), yerr=np.std(res, axis=0),
            label=lbl, color=color, linestyle='--' if reinf else '-',  alpha=0.7 if reinf else 1.0)
    else:
        plt.plot(
            x=np.arange(ndots) * dot_every, y=np.mean(res, axis=0),
            label=lbl, color=color, linestyle='--' if reinf else '-', alpha=0.7 if reinf else 1.0)

ax = plt.gca()
ax.set_ylim(bottom=0)
ax.set_xlabel('iterations')
ax.set_ylabel('mean-squared error')
ax.legend()

util.basic()

plt.savefig('./paper/identity/identity.pdf', dpi=600)