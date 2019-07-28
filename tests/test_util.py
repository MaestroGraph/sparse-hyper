import _context

import unittest
import torch, sys
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
#import sparse.layers.densities

import util, sys

class TestLayers(unittest.TestCase):

    def test_unique(self):
        r = util.unique(torch.tensor( [[1,2,3,4],[4,3,2,1],[1,2,3,4]] ))

        self.assertEqual((3, 1), r.size())
        self.assertEqual(r[0], r[2])
        self.assertNotEqual(r[0], r[1])
        self.assertNotEqual(r[1], r[2])

        r = util.nunique(torch.tensor( [[[1,2,3,4],[4,3,2,1],[1,2,3,4]]] ))

        self.assertEqual((1, 3), r.size())
        self.assertEqual(r[0, 0], r[0, 2])
        self.assertNotEqual(r[0, 0], r[0, 1])
        self.assertNotEqual(r[0, 1], r[0, 2])

    def test_duplicates(self):

        tuples = torch.tensor([
                [[5, 5], [1, 1], [2, 3], [1, 1]],
                [[3, 2], [3, 2], [5, 5], [5, 5]]
            ])
        dedup = torch.tensor([
                [[5, 5], [1, 1], [2, 3], [0, 0]],
                [[3, 2], [0, 0], [5, 5], [0, 0]]
            ])


        dup = util.duplicates(tuples)
        tuples[dup, :] = tuples[dup, :] * 0
        self.assertEqual( (tuples != dedup).sum(), 0)

        tuples = torch.tensor([[
                [3, 1],
                [3, 2],
                [3, 1],
                [0, 3],
                [0, 2],
                [3, 0],
                [0, 3],
                [0, 0]]])

        self.assertEqual([0, 0, 1, 0, 0, 0, 1, 0], list(util.duplicates(tuples).view(-1)))

    def test_nduplicates(self):

        # some tuples
        tuples = torch.tensor([
            [[5, 5], [1, 1], [2, 3], [1, 1]],
            [[3, 2], [3, 2], [5, 5], [5, 5]]
        ])

        # what they should look like after masking out the duplicates
        dedup = torch.tensor([
            [[5, 5], [1, 1], [2, 3], [0, 0]],
            [[3, 2], [0, 0], [5, 5], [0, 0]]
        ])

        # add a load of dimensions
        tuples = tuples[None, None, None, :, :, :].expand(3, 5, 7, 2, 4, 2).contiguous()
        dedup  = dedup[None, None, None, :, :, :].expand(3, 5, 7, 2, 4, 2).contiguous()

        # find the duplicates
        dup = util.nduplicates(tuples)

        # mask them out
        tuples[dup, :] = tuples[dup, :] * 0
        self.assertEqual((tuples != dedup).sum(), 0) # assert equal to expected

        # second test: explicitly test the bitmask returned by nduplicates
        tuples = torch.tensor([[
            [3, 1],
            [3, 2],
            [3, 1],
            [0, 3],
            [0, 2],
            [3, 0],
            [0, 3],
            [0, 0]]])

        tuples = tuples[None, None, None, :, :, :].expand(8, 1, 7, 1, 8, 2).contiguous()

        self.assertEqual([0, 0, 1, 0, 0, 0, 1, 0], list(util.nduplicates(tuples)[0, 0, 0, :, :].view(-1)))

        # third test: single element tuples
        tuples = torch.tensor([
            [[5], [1], [2], [1]],
            [[3], [3], [5], [5]]
        ])
        dedup = torch.tensor([
            [[5], [1], [2], [0]],
            [[3], [0], [5], [0]]
        ])

        tuples = tuples[None, None, None, :, :, :].expand(3, 5, 7, 2, 4, 2).contiguous()
        dedup  = dedup[None, None, None, :, :, :].expand(3, 5, 7, 2, 4, 2).contiguous()

        dup = util.nduplicates(tuples)

        tuples[dup, :] = tuples[dup, :] * 0
        self.assertEqual((tuples != dedup).sum(), 0)

    def test_nduplicates_recursion(self):
        """
        Reproducing observed recursion error
        :return:
        """

        # tensor of 6 1-tuples
        tuples = torch.tensor(
            [[[[74],
               [75],
               [175],
               [246],
               [72],
               [72]]]])

        dedup = torch.tensor(
            [[[[74],
               [75],
               [175],
               [246],
               [72],
               [0]]]])

        dup = util.nduplicates(tuples)

        tuples[dup, :] = tuples[dup, :] * 0
        self.assertEqual((tuples != dedup).sum(), 0)

    def test_unique_recursion(self):
        """
        Reproducing observed recursion error
        :return:
        """

        # tensor of 6 1-tuples
        tuples = torch.tensor(
            [[74],
               [75],
               [175],
               [246],
               [72],
               [72]])
        dup = util.unique(tuples)

    def test_wrapmod(self):

        self.assertAlmostEqual(util.wrapmod(torch.tensor([9.1]), 9).item(), 0.1, places=5)

        self.assertAlmostEqual(util.wrapmod(torch.tensor([-9.1]), 9).item(), 8.9, places=5)

        self.assertAlmostEqual(util.wrapmod(torch.tensor([-0.1]), 9).item(), 8.9, places=5)

        self.assertAlmostEqual(util.wrapmod(torch.tensor([10.0, -9.1]), 9)[1].item(), 8.9, places=5)

    def test_interpolation_grid(self):

        g = util.interpolation_grid()
        self.assertEqual( (torch.abs(g.sum(dim=2) - 1.0) > 0.0001).sum(), 0)

        g = util.interpolation_grid((3, 3))
        self.assertAlmostEqual(g[0, 0, 0], 1.0, 5)
        self.assertAlmostEqual(g[0, 2, 1], 1.0, 5)
        self.assertAlmostEqual(g[2, 2, 2], 1.0, 5)
        self.assertAlmostEqual(g[2, 0, 3], 1.0, 5)

    def test_coordconv(self):

        """
        Performs the regression experiment from https://arxiv.org/abs/1807.03247

        Tests whether the coord_conv layer behaves as expected. Baseline false should not converge to zero, baseline true
        should.

        :return:
        """

        RES = 64
        B, H, W = 16, RES, RES
        BATCHES = 16000

        for baseline in (False, True):
            print('baseline', baseline)

            C = nn.Conv2d if baseline else util.CConv2d

            model = nn.Sequential(
                C(1, 8, stride=1, kernel_size=1, padding=0),
                #nn.ReLU(),
                nn.Conv2d(8, 8, stride=1, kernel_size=1, padding=0),
                #nn.ReLU(),
                nn.Conv2d(8, 8, stride=1, kernel_size=1, padding=0),
                #nn.ReLU(),
                nn.Conv2d(8, 8, stride=1, kernel_size=3, padding=1),
                #nn.ReLU(),
                nn.Conv2d(8, 2, stride=1, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=(RES, RES)),
                util.Lambda(lambda x : x.squeeze())
            )

            opt = torch.optim.Adam(lr=0.005, params=model.parameters())

            for it in range(BATCHES):

                # generate data

                target = (torch.rand(size=(B, 2)) * RES).to(torch.long)

                hf = RES // 2

                test_indices = (target[:, 0] < hf) & (target[:, 1] < hf)

                target_train = target[~ test_indices, :]
                target_test  = target[  test_indices, :]

                input_train = torch.zeros(target_train.size(0), H, W)
                input_test  = torch.zeros( target_test.size(0), H, W)

                input_train[torch.arange(input_train.size(0), dtype=torch.long), target_train[:, 0], target_train[:, 1]] = 1
                input_test [torch.arange(input_test.size(0),  dtype=torch.long), target_test [:, 0], target_test [:, 1]] = 1

                input_train = input_train[:, None, :, :]
                input_test = input_test[:, None, :, :]

                input_train, input_test, target_train, target_test = Variable(input_train), Variable(input_test), Variable(target_train), Variable(target_test)


                opt.zero_grad()
                # list(model.modules())[1].master.weight.retain_grad()

                out = model(input_train)
                loss = F.mse_loss(out, target_train.to(torch.float))

                loss.backward()
                opt.step()

                if it % 2000 == 0:
                    with torch.no_grad():
                        out = model(input_test)
                        tloss = F.mse_loss(out, target_test.to(torch.float))

                        print(f'train loss at {it}: { loss.item():.04}')
                        print(f' test loss at {it}: {tloss.item():.04}')

                        # print(list(model.modules())[1].master.weight.grad)

if __name__ == '__main__':
    unittest.main()