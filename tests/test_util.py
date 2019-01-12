import _context

import unittest
import torch, sys
#import sparse.layers.densities

import util

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

        tuples = torch.tensor([
            [[5, 5], [1, 1], [2, 3], [1, 1]],
            [[3, 2], [3, 2], [5, 5], [5, 5]]
        ])
        dedup = torch.tensor([
            [[5, 5], [1, 1], [2, 3], [0, 0]],
            [[3, 2], [0, 0], [5, 5], [0, 0]]
        ])

        tuples = tuples[None, None, None, :, :, :].expand(3, 5, 7, 2, 4, 2).contiguous()
        dedup  = dedup[None, None, None, :, :, :].expand(3, 5, 7, 2, 4, 2).contiguous()

        dup = util.nduplicates(tuples)

        tuples[dup, :] = tuples[dup, :] * 0
        self.assertEqual((tuples != dedup).sum(), 0)

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

    def test_wrapmod(self):

        self.assertAlmostEqual(util.wrapmod(torch.tensor([9.1]), 9).item(), 0.1, places=5)

        self.assertAlmostEqual(util.wrapmod(torch.tensor([-9.1]), 9).item(), 8.9, places=5)

        self.assertAlmostEqual(util.wrapmod(torch.tensor([-0.1]), 9).item(), 8.9, places=5)

        self.assertAlmostEqual(util.wrapmod(torch.tensor([10.0, -9.1]), 9)[1].item(), 8.9, places=5)


if __name__ == '__main__':
    unittest.main()