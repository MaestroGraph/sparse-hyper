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

if __name__ == '__main__':
    unittest.main()