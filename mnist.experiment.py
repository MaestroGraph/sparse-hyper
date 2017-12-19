import hyper, gaussian, util, logging, time, pretrain
import torch, random
from torch.autograd import Variable
from torch import nn, optim
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from util import Lambda, Debug

from torchsample.metrics import CategoricalAccuracy

import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from util import od

from argparse import ArgumentParser

"""
MNIST experiment

"""

def go(batch=64, epochs=350, k=750, additional=512, model='baseline', cuda=False, seed=1, pretrain_lr=0.001,
       pretrain_epochs=20, bias=True, data='./data', lr=0.01):
    torch.manual_seed(seed)
    logging.basicConfig(filename='run.log',level=logging.INFO)
    LOG = logging.getLogger()

    l1Params = None
    L1WEIGHT = 0.1

    w = SummaryWriter()

    SHAPE = (1, 28, 28)

    gaussian.PROPER_SAMPLING = False

    normalize = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=normalize)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=2)
    test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=normalize)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

    if model == 'non-adaptive':
        shapes = [SHAPE, (4, 16, 16), (8, 8, 8)]
        layers = [
            gaussian.ParamASHLayer(shapes[0], shapes[1], k=k, additional=additional, has_bias=bias),
            nn.Sigmoid(),
            gaussian.ParamASHLayer(shapes[1], shapes[2], k=k, additional=additional, has_bias=bias),
            nn.Sigmoid(),
            util.Flatten(),
            nn.Linear(512, 10),
            nn.Softmax()]
        pivots = [2, 4]
        decoder_channels = [True, True]

        pretrain.pretrain(layers, shapes, pivots, trainloader, epochs=pretrain_epochs, k_out=k, out_additional=additional, use_cuda=cuda,
                plot=True, out_has_bias=bias, has_channels=decoder_channels, learn_rate=pretrain_lr)

        model = nn.Sequential(od(layers))

        if cuda:
            model.apply(lambda t: t.cuda())

    elif model == 'free':

        shapes = [SHAPE, (4, 16, 16), (8, 8, 8)]
        layers = [
            gaussian.CASHLayer(shapes[0], shapes[1], k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False),
            nn.Sigmoid(),
            gaussian.CASHLayer(shapes[1], shapes[2], k=k, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False),
            nn.Sigmoid(),
            util.Flatten(),
            nn.Linear(512, 10),
            nn.Softmax()]
        pivots = [2, 4]
        decoder_channels = [True, True]

        pretrain.pretrain(layers, shapes, pivots, trainloader, epochs=pretrain_epochs, k_out=k, out_additional=additional, use_cuda=cuda,
                plot=True, out_has_bias=bias, has_channels=decoder_channels, learn_rate=pretrain_lr)

        model = nn.Sequential(od(layers))

        if cuda:
            model.apply(lambda t: t.cuda())

    elif model == 'free9':

        shapes = [SHAPE, (4, 16, 16), (8, 8, 8)]
        layers = [
            gaussian.CASHLayer(shapes[0], shapes[1], k=k, ksize=9, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False),
            nn.Sigmoid(),
            gaussian.CASHLayer(shapes[1], shapes[2], k=k, ksize=9, additional=additional, has_bias=bias, has_channels=True, adaptive_bias=False),
            nn.Sigmoid(),
            util.Flatten(),
            nn.Linear(512, 10),
            nn.Softmax()]
        pivots = [2, 4]
        decoder_channels = [True, True]

        pretrain.pretrain(layers, shapes, pivots, trainloader, epochs=pretrain_epochs, k_out=k, out_additional=additional, use_cuda=cuda,
                plot=True, out_has_bias=bias, has_channels=decoder_channels, learn_rate=pretrain_lr)

        model = nn.Sequential(od(layers))

        if cuda:
            model.apply(lambda t: t.cuda())

    elif model == 'baseline':
        model = nn.Sequential(
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=3),
            nn.MaxPool2d(stride=2, kernel_size=2),
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(stride=2, kernel_size=2),
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(stride=2, kernel_size=2),
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            util.Flatten(),
            nn.Linear(256, 10),
            nn.Softmax())

        if cuda:
            model = model.cuda()

    elif model == 'baseline-big':

        # Reload the data with augmentation
        normalize = transforms.Compose([transforms.RandomCrop(size=28, padding=2), transforms.ToTensor()])
        train = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=normalize)
        trainloader = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=2)
        test = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=normalize)
        testloader = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False, num_workers=2)

        model = nn.Sequential(
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(stride=2, kernel_size=2),
            #  Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(stride=2, kernel_size=2),
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # nn.MaxPool2d(stride=2, kernel_size=2),
            # Debug(lambda x: print(x.size(), util.prod(x[-1:].size()))),
            util.Flatten(),
            nn.Linear(100352, 328),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(328, 196),
            nn.Softmax())

        print(util.count_params(model), ' parameters')

        if cuda:
            model = model.cuda()

    elif model == 'baseline-dense':

        lin1 = nn.Linear(28*28, 4*16*16)
        lin2 = nn.Linear(4*16*16, 8*8*8)

        model = nn.Sequential(
            util.Flatten(),
            lin1,
            nn.ReLU(),
            lin2,
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax()
        )

        l1Params = [lin1.weight, lin2.weight]

        if cuda:
            model = model.cuda()

    else:
        raise Exception('Model {} not found'.format(model))

    ## SIMPLE
    criterion = nn.CrossEntropyLoss()
    acc = CategoricalAccuracy()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    step = 0

    for epoch in range(epochs):
        for i, data in tqdm(enumerate(trainloader, 0)):

            # get the inputs
            inputs, labels = data
            # inputs = inputs.squeeze(1)

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if l1Params is not None:
                for l1_param in l1Params:
                    loss += L1WEIGHT * l1_param.abs().sum()

            t0 = time.time()
            loss.backward()  # compute the gradients
            logging.info('backward: {} seconds'.format(time.time() - t0))
            optimizer.step()

            w.add_scalar('mnist/train-loss', loss.data[0], step)

            step += 1

        total = 0.0
        num = 0
        for i, data in enumerate(testloader, 0):
            # get the inputs
            inputs, labels = data
            # inputs = inputs.squeeze(1)

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            # wrap them in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)

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

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.01, type=float)

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
       model=options.model, cuda=options.cuda, pretrain_epochs=options.pretrain_epochs, data=options.data, lr=options.lr )
