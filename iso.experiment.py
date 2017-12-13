import hyper, gaussian, util, logging, time, pretrain
import torch, random
from torch.autograd import Variable
from torch import nn, optim
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from util import Lambda, Debug

from torch.utils.data import TensorDataset, DataLoader

from torchsample.metrics import CategoricalAccuracy

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from util import od, prod

from argparse import ArgumentParser

import networkx as nx

"""
MNIST experiment

"""

def generate(n=128, m=512, num=64, cuda=False):
    FT = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LT = torch.cuda.LongTensor if cuda else torch.LongTensor


    data = FT(num, 2, n, n)
    classes = LT(num)

    for i in range(num):
        graph1 = nx.gnm_random_graph(n, m)
        am1 = nx.to_numpy_matrix(graph1)

        if random.choice([True, False]):
            # graphs are isomorphic
            nodes = graph1.nodes()
            random.shuffle(nodes)

            am2 = nx.to_numpy_matrix(graph1, nodelist=nodes)

            classes[i] = 0
        else:
            # graphs are (probably not) isomorphic
            graph2 = nx.gnm_random_graph(n, m)
            am2 = nx.to_numpy_matrix(graph2)

            classes[i] = 1

        data[i, 0, :, :] = torch.from_numpy(am1)
        data[i, 1, :, :] = torch.from_numpy(am2)

    return data, classes

TRAIN_SIZE = 60000
TEST_SIZE = 15000

def go(nodes=128, links=512, batch=64, epochs=350, k=750, additional=512, modelname='baseline', cuda=False, seed=1, pretrain_lr=0.001, pretrain_epochs=20, bias=True, data='./data'):

    torch.manual_seed(seed)
    logging.basicConfig(filename='run.log',level=logging.INFO)
    LOG = logging.getLogger()

    w = SummaryWriter()

    SHAPE = (1, nodes, nodes)

    print('generating data...')
    train, train_labels = generate(nodes, links, TRAIN_SIZE, cuda=cuda)
    test, test_labels = generate(nodes, links, TRAIN_SIZE, cuda=cuda)
    print('done.')

    ds_train = TensorDataset(train, train_labels)
    ds_test = TensorDataset(test, test_labels)

    train_loader = DataLoader(ds_train,batch_size=batch,shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=batch, shuffle=True)

    if modelname == 'non-adaptive':
        shapes = [SHAPE, (4, 32, 32), (8, 4, 4)]
        layers = [
            gaussian.ParamASHLayer(shapes[0], shapes[1], k=k, additional=additional, has_bias=bias),
            nn.Sigmoid(),
            gaussian.ParamASHLayer(shapes[1], shapes[2], k=k, additional=additional, has_bias=bias),
            nn.Sigmoid(),
            util.Flatten(),
            nn.Linear(128, 32),
            nn.Sigmoid()]
        pivots = [2, 4]
        decoder_channels = [True, True]

        pretrain.pretrain(layers, shapes, pivots, train_loader, epochs=pretrain_epochs, k_out=k, out_additional=additional, use_cuda=cuda,
                plot=True, out_has_bias=bias, has_channels=decoder_channels, learn_rate=pretrain_lr)

        model = nn.Sequential(od(layers))

        if cuda:
            model.apply(lambda t: t.cuda())

    elif modelname == 'free':

        shapes = [SHAPE, (4, 32, 32), (8, 4, 4)]
        layers = [
            gaussian.CASHLayer(shapes[0], shapes[1], k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False),
            nn.Sigmoid(),
            gaussian.CASHLayer(shapes[1], shapes[2], k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False),
            nn.Sigmoid(),
            util.Flatten(),
            nn.Linear(128, 32),
            nn.Sigmoid()]
        pivots = [2, 4]
        decoder_channels = [True, True]

        pretrain.pretrain(layers, shapes, pivots, train_loader, epochs=pretrain_epochs, k_out=k, out_additional=additional, use_cuda=cuda,
                plot=True, out_has_bias=bias, has_channels=decoder_channels, learn_rate=pretrain_lr)

        model = nn.Sequential(od(layers))

        if cuda:
            model.apply(lambda t: t.cuda())

    elif modelname == 'baseline':
        model = nn.Sequential(
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(stride=4, kernel_size=4),
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))), # (4, 32, 32)
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(stride=4, kernel_size=4),
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))), # (8, 8, 8)
            # nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=1, padding=2),
            # nn.MaxPool2d(stride=4, kernel_size=4),
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            util.Flatten(),
            nn.Linear(512, 32),
            nn.Sigmoid())

        if cuda:
            model = model.cuda()

    elif modelname == 'baseline-big':
        model = nn.Sequential(
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(stride=2, kernel_size=2),
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))), # (4, 32, 32)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(stride=2, kernel_size=2),
            #Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))), # (64, 8, 8)

            util.Flatten(),
            nn.Linear(1024, 32),
            nn.Sigmoid())

        if cuda:
            model = model.cuda()

    decoder = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
        nn.Softmax()
    )

    ## SIMPLE
    criterion = nn.CrossEntropyLoss()
    acc = CategoricalAccuracy()
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=0.01)

    step = 0

    for epoch in range(epochs):
        for i, data in tqdm(enumerate(train_loader, 0)):

            # get the inputs
            graphs, labels = data
            graphs1, graphs2 = graphs[:, 0, :], graphs[:, 1, :]

            graphs1, graphs2, labels = Variable(graphs1), Variable(graphs2), Variable(labels)
            graphs1, graphs2 = graphs1.unsqueeze(1), graphs2.unsqueeze(1)

            # forward + backward + optimize
            optimizer.zero_grad()
            rep1 = model(graphs1)
            rep2 = model(graphs2)

            rep = torch.cat((rep1, rep2), dim=1)

            outputs = decoder(rep)

            # print(outputs, labels)
            loss = criterion(outputs, labels)

            t0 = time.time()
            loss.backward()  # compute the gradients
            logging.info('backward: {} seconds'.format(time.time() - t0))
            optimizer.step()

            w.add_scalar('graphs/train-loss', loss.data[0], step)

            step += 1

        total = 0.0
        num = 0
        for i, data in tqdm(enumerate(test_loader, 0)):

            # get the inputs
            graphs, labels = data
            graphs1, graphs2 = graphs[:, 0, :], graphs[:, 1, :]

            graphs1, graphs2, labels = Variable(graphs1), Variable(graphs2), Variable(labels)
            graphs1, graphs2 = graphs1.unsqueeze(1), graphs2.unsqueeze(1)

            # forward + backward + optimize
            optimizer.zero_grad()
            rep1 = model(graphs1)
            rep2 = model(graphs2)

            rep = torch.cat((rep1, rep2), dim=1)

            outputs = decoder(rep)

            total += acc(outputs, labels)
            num += 1

        accuracy = total/num

        w.add_scalar('mnist/per-epoch-test-acc', accuracy, epoch)
        print('EPOCH {}: {} accuracy '.format(epoch, accuracy))

    print('Finished Training.')

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-m", "--model",
                        dest="model",
                        help="Which model to train.",
                        default='baseline')

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples",
                        default=750, type=int)

    parser.add_argument("-a", "--additional",
                        dest="additional",
                        help="Number of additional points sampled",
                        default=512, type=int)

    parser.add_argument("-p.e", "--pretrain-epochs",
                        dest="pretrain_epochs",
                        help="Number of training epochs per layer",
                        default=20, type=int)

    parser.add_argument("-p.l", "--pretrain-learn-rate",
                        dest="plr",
                        help="Pretraining learn rate",
                        default=0.001, type=float)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("-B", "--no-bias", dest="bias",
                        help="Whether to give the layers biases.",
                        action="store_false")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data directory",
                        default='./data')

    options = parser.parse_args()

    print('OPTIONS', options)

    go(batch=options.batch_size, k=options.k, pretrain_lr=options.plr, bias=options.bias, additional=options.additional,
       modelname=options.model, cuda=options.cuda, pretrain_epochs=options.pretrain_epochs, data=options.data )
