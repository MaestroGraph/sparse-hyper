import torch
import torch.nn.functional as F
from torch import nn

import sparse.util
import numpy as np

"""
Modules to implement differentiable quicksort. ```Split``` implements the half-permutation, and ```SortLayer``` chains
these into quicksort.
"""

class Split(nn.Module):
    """
    A split matrix moves the elements of the input to either the top or the bottom
    half of a subsection of the output, but keeps the ordering intact otherwise.

    For depth 0, each element is moved to the top or bottom half of the output. For
    depth 1 each element is moved to the top or bottom half of its current half of
    the matrix and so on.

    """
    def __init__(self, size, depth, additional=1, sigma_scale=0.1, sigma_floor=0.0):
        super().__init__()

        template = torch.LongTensor(range(size)).unsqueeze(1).expand(size, 2)
        self.register_buffer('template', template)

        self.size = size
        self.depth = depth
        self.sigma_scale = sigma_scale
        self.sigma_floor = sigma_floor
        self.additional = additional

    def duplicates(self, tuples):
        """
        Takes a list of tuples, and for each tuple that occurs mutiple times
        marks all but one of the occurences (in the mask that is returned).

        :param tuples: A size (batch, k, rank) tensor of integer tuples
        :return: A size (batch, k) mask indicating the duplicates
        """
        b, k, r = tuples.size()

        # unique = ((tuples.float() + 1) ** primes).prod(dim=2)  # unique identifier for each tuple
        unique = util.unique(tuples.view(b*k, r)).squeeze().view(b, k)

        sorted, sort_idx = torch.sort(unique, dim=1)
        _, unsort_idx = torch.sort(sort_idx, dim=1)

        mask = sorted[:, 1:] == sorted[:, :-1]
        # mask = mask.view(b, k - 1)

        zs = torch.zeros(b, 1, dtype=torch.uint8, device='cuda' if tuples.is_cuda else 'cpu')
        mask = torch.cat([zs, mask], dim=1)

        return torch.gather(mask, 1, unsort_idx)

    def generate_integer_tuples(self, offset, additional=16):

        b, s = offset.size()

        choices = offset.round().byte()[:, None, :]

        if additional > 0:
            sampled = util.sample_offsets(b, additional, s, self.depth, cuda=offset.is_cuda)
            # sampled = ~ choices

            choices = torch.cat([choices, sampled], dim=1).byte()

        return self.generate(choices, offset)

    def generate(self, choices, offset):

        b, n, s = choices.size()

        offset = offset[:, None, :].expand(b, n, s)

        probs = offset.clone()
        probs[~ choices] = 1.0 - probs[~ choices]
        # prob now contains the probability (under offset) of the choices made
        probs = probs.prod(dim=2, keepdim=True).expand(b, n, s).contiguous()

        # Generate indices from the chosen offset
        indices = util.split(choices, self.depth)

        if n > 1:
            dups = self.duplicates(indices)

            probs = probs.clone()
            probs[dups] = 0.0

        probs = probs / probs.sum(dim=1, keepdim=True)

        return indices, probs

    def forward(self, input, keys, offset, train=True, reverse=False, verbose=False):

        if train:
            indices, probs = self.generate_integer_tuples(offset, self.additional)
        else:
            indices, probs = self.generate_integer_tuples(offset, 0)

        if verbose:
            print(indices[0, 0])

        indices = indices.detach()
        b, n, s = indices.size()

        template = self.template[None, None, :, :].expand(b, n, s, 2).contiguous()
        if not reverse: # normal half-permutation
            template[:, :, :, 0] = indices
        else: # reverse the permutation
            template[:, :, :, 1] = indices
        indices = template

        indices = indices.contiguous().view(b, -1, 2)
        probs = probs.contiguous().view(b, -1)

        output   = util.batchmm(indices, probs, (s, s), input)

        keys_out = util.batchmm(indices, probs, (s, s), keys[:, :, None]).squeeze()

        return output, keys_out

class SortLayer(nn.Module):
    """

    """
    def __init__(self, size, additional=0, sigma_scale=0.1, sigma_floor=0.0, certainty=10.0):
        super().__init__()

        mdepth = int(np.log2(size))

        self.layers = nn.ModuleList()
        for d in range(mdepth):
            self.layers.append(Split(size, d, additional, sigma_scale, sigma_floor))

        # self.certainty = nn.Parameter(torch.tensor([certainty]))
        self.register_buffer('certainty', torch.tensor([certainty]))

        # self.offset = nn.Sequential(
        #     util.Lambda(lambda x : x[:, 0] - x[:, 1]),
        #     util.Lambda(lambda x : x * self.certainty),
        #     nn.Sigmoid()
        # )

    def forward(self, x, keys, target=None, train=True, verbose=False):

        xs = [x]
        targets = [target]
        offsets = []

        b, s, z = x.size()
        b, s = keys.size()

        t = target

        for d, split in enumerate(self.layers):

            buckets = keys[:, :, None].view(b, 2**d, -1)

            # compute pivots
            pivots = buckets.view(b*2**d, -1)
            pivots = median(pivots, keepdim=True)
            pivots = pivots.view(b, 2 ** d, -1).expand_as(buckets)

            pivots = pivots.contiguous().view(b, -1).expand_as(keys)

            # compute offsets by comparing values to pivots
            if train:
                offset = keys - pivots
                offset = F.sigmoid(offset * self.certainty)
            else:
                offset = (keys > pivots).float()

            # offset = offset.round() # DEBUG
            offsets.append(offset)


            x, keys = split(x, keys, offset, train=train, verbose=verbose)
            xs.append(x)

            if verbose:
                print('o', offset[0])
                print('k', keys[0])


        if target is not None:
            for split, offset in zip(self.layers[::-1], offsets[::-1]):
                t, _ = split(t, keys, offset, train=train, reverse=True)
                targets.insert(0, t)

        if target is None:
            return x, keys

        return xs, targets, keys

def median(x, keepdim=False):
    b, s = x.size()

    y = x.sort(dim=1)[0][:, s//2-1:s//2+1].mean(dim=1, keepdim=keepdim)

    return y

if __name__ == '__main__':

    x = torch.randn(3, 4)
    print(x)
    print(median(x))