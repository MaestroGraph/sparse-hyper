import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

batch = 64
iterations = 50000

# Two layer MLP, producing means and sigmas for the output
h = 128
model = nn.Sequential(
    nn.Linear(1, h), nn.Sigmoid(),
    nn.Linear(h, 8)
)

opt = torch.optim.Adam(model.parameters(), lr = 0.0005)

for i in range(iterations):

    x = torch.randn(batch, 1)
    y = torch.cat([x, x ** 2, x ** 3, torch.sin(x)], dim=1)

    x, y = Variable(x), Variable(y)

    res = model(x)
    means, sigs = res[:, :4], torch.exp(res[:, 4:])

    dists = torch.distributions.Normal(means, sigs)
    samples = dists.sample()

    # REINFORCE
    mloss = F.mse_loss(samples, y, reduce=False)
    loss = - dists.log_prob(samples) * - mloss
    loss = loss.mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % 1000 == 0:
        print('{: 6} grad'.format(i), list(model.parameters())[0].grad.mean())
        print('      ', 'loss', F.mse_loss(samples.data, y.data, reduce=False).mean(dim=0))
        print('      ', 'sigs', sigs.mean(dim=0))

