import torch

from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import trange

import util, logging, time, gc
logging.basicConfig(filename='run.log',level=logging.INFO)
LOG = logging.getLogger()


from argparse import ArgumentParser


def go(arg):

    b, s = arg.batch_size, arg.size

    tomean = nn.Linear(s, 1, bias=False)
    compare = nn.Linear(2, 1, bias=False)
    scale = nn.Parameter(torch.ones(1))
    params = list(tomean.parameters()) + list(compare.parameters()) + [scale]

    models = [tomean, compare, scale]

    if arg.cuda:
        for model in models:
            model.cuda()

    opt = torch.optim.Adam(params, lr=arg.lr)

    for i in trange(arg.iterations):

        x = torch.randn(b, s)
        t = (x > x.mean(dim=1, keepdim=True)).float().expand(b, s)

        if arg.cuda:
            x, t = x.cuda, t.cuda

        x, t = Variable(x), Variable(t)

        opt.zero_grad()

        means = tomean(x).expand(b, s)
        comps = compare(torch.cat([x.view(-1, 1), means.contiguous().view(-1, 1)], dim=1))
        comps = comps.view(b, s)
        comps = torch.sigmoid(comps * scale)

        loss = F.mse_loss(comps, t)

        loss.backward()
        opt.step()

        if i % 5000 == 0:
            print(i, loss.item())
            print(scale)
            print(compare.weight)
            print(tomean.weight)

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-s", "--size",
                        dest="size",
                        help="Dimensionality of the input.",
                        default=8, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=128, type=int)

    parser.add_argument("-i", "--iterations",
                        dest="iterations",
                        help="Number of iterations (in batches).",
                        default=8000, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="Random seed.",
                        default=32, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)
    LOG.info('OPTIONS ' + str(options))

    go(options)
