import _context

import unittest
import torch
#import sparse.layers.densities

import layers

class TestLayers(unittest.TestCase):

    def test_densities(self):

        means  = torch.tensor([[[0.0]]])
        sigmas = torch.tensor([[[1.0]]])
        points = torch.tensor([[[0.0]]])

        density = layers.densities(points, means, sigmas)
        self.assertAlmostEqual(1.0, density, places=7)

        means  = torch.tensor([[[0.0, 0.0], [1.0, 1.0]],[[2.0, 2.0], [4.0, 4.0]]])
        sigmas = torch.tensor([[[1.0, 1.0], [1.0, 1.0]],[[1.0, 1.0], [1.0, 1.0]]])
        points = torch.tensor([[[0.0, 0.0], [1.0, 1.0]],[[2.0, 2.0], [4.0, 4.0]]])

        density = layers.densities(points, means, sigmas)

        self.assertEquals((2, 2, 2), density.size())
        self.assertAlmostEqual(1.0, density[0, 0, 0], places=7)

        means  = torch.randn(3, 5, 16)
        sigmas = torch.randn(3, 5, 16).abs()
        points = torch.randn(3, 7, 16)

        density = layers.densities(points, means, sigmas)

        self.assertEquals((3, 7, 5), density.size())

    def test_ngenerate(self):

        means  = torch.randn(6, 2, 3)
        sigmas = torch.randn(6, 2)
        values = torch.randn(6, 2)

        b = 5
        size = (64, 128, 32)

        ms = means.size()

        xp = (5, ) + means.size()
        # Expand parameters along batch dimension
        means = means.expand(*xp)
        sigmas = sigmas.expand(*xp[:-1])
        values = values.expand(*xp[:-1])

        means, sigmas = layers.transform_means(means, size), layers.transform_sigmas(sigmas, size)

        indices_old = layers.generate_integer_tuples(means,
                                   2, 2,
                                   relative_range=(4, 4, 4),
                                   rng=size,
                                   cuda=means.is_cuda)


        indices_new = layers.ngenerate(means,
                                   2, 2,
                                   relative_range=(4, 4, 4),
                                   rng=size,
                                   cuda=means.is_cuda)

        assert indices_old.size() == indices_new.size()

        for i in range(indices_new.view(-1, 3).size(0)):
            print(indices_new.view(-1, 3)[i])

    def test_conv(self):

        x = torch.ones(1, 4, 3, 3)

        c = layers.Convolution((4, 3, 3), 4, k=2, rprop=.5, gadditional=2, radditional=2)

        print(c(x))

if __name__ == '__main__':
    unittest.main()