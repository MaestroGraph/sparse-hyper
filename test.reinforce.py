import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

batch = 64
iterations = 100000
reinforce = True

# Two layer MLP, producing means and sigmas for the output
h = 128
model = nn.Sequential(
    nn.Linear(1, h), nn.ReLU(),
    nn.Linear(h, h), nn.ReLU(),
    nn.Linear(h, 8)
)

tomean = nn.Linear(4, 1, bias=False)

opt = torch.optim.Adam(list(model.parameters()) + list(tomean.parameters()), lr = 0.0001)

for i in range(iterations):

    x = torch.randn(batch, 1)
    t = torch.cat([x, x ** 2, x ** 3, torch.sin(x)], dim=1).mean(dim=1)

    x, t = Variable(x), Variable(t)

    res = model(x)
    means, sigs = res[:, :4], torch.exp(res[:, 4:])

    dists = torch.distributions.Normal(means, sigs)
    samples = dists.sample()

    if reinforce:
        y = tomean(samples).squeeze()
    else:
        y = tomean(means).squeeze()

    if reinforce:
        mloss = F.mse_loss(y, t, reduce=False)[:, None].expand_as(samples)

        loss = - dists.log_prob(samples) * - mloss.detach()  # REINFORCE
        loss = (loss + mloss).mean()
    else:
        loss = F.mse_loss(y, t)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if not i % 1000:
        print('{: 6} tomean'.format(i), tomean.weight.data, tomean.weight.grad)
        print('      ', 'grad'.format(i), list(model.parameters())[0].grad.mean())
        print('      ', 'loss', F.mse_loss(y.data, t.data, reduce=False).mean(dim=0))
        if reinforce:
            print('      ', 'sigs', sigs.mean(dim=0))

