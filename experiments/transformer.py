from _context import sparse
from sparse import util

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from util import d

# import warnings
# warnings.simplefilter("error")
# warnings.simplefilter("ignore", DeprecationWarning)

# from util import tic, toc

# NB, the enwik8 data contains tokens from 9 to 240
NUM_TOKENS = 256
LOG2E = math.log2(math.e)
MARGIN = 0.1

def sample(lnprobs, temperature=1.0):

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

class MSparseSelfAttention(nn.Module):
    """
    Masked sparse self attention (two degrees of freedom)
    """
    def __init__(self, emb, k, gadditional, radditional, region, heads=8, mask=False, min_sigma=0.05, sigma_scale=0.1):
        """

        :param emb:
        :param k: Number of connections to the input in total
        :param gadditional:
        :param radditional:
        :param region:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb, self.heads, self.mask, self.min_sigma, self.sigma_scale = emb, heads, mask, min_sigma, sigma_scale

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

        self.gadditional = gadditional
        self.radditional = radditional
        self.region = region
        self.k = k

        self.means  = nn.Parameter(torch.randn((k, 2)))
        self.sigmas = nn.Parameter(torch.randn((k, )))
        self.register_buffer('mvalues', torch.ones((k, )))

    def hyper(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region

        # generate the continuous parameters
        means = self.means[None, None, :, :].expand(b, 1, k, 2)
        sigmas = self.sigmas[None, None, :].expand(b, 1, k)
        values = self.mvalues[None, None, :].expand(b, 1, k)

        means = util.flip(means.contiguous())  # flip everything to below the diagonal of the matrix

        s = (t, t)
        means, sigmas = sparse.transform_means(means, s), \
                        sparse.transform_sigmas(sigmas, s, min_sigma=self.min_sigma) * self.sigma_scale

        return means, sigmas, values

    def forward(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region
        s = (t, t)

        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        means, sigmas, mvalues = self.hyper(x)

        # sample integer indices and values
        indices = sparse.ngenerate(means, self.gadditional, self.radditional, rng=(t, t), relative_range=(self.region, self.region), cuda=x.is_cuda)
        indices = util.flip(indices)

        indfl = indices.float()

        vs = k * (4 + self.radditional + self.gadditional)
        assert indices.size() == (b, 1, vs, 2), f'{indices.size()}, {(b, 1, vs, 2)}'

        # Mask for duplicate indices
        dups = util.nduplicates(indices).to(torch.bool)

        # compute (unnormalized) densities under the given MVNs (proportions)
        props = sparse.densities(indfl, means, sigmas).clone()
        props[dups, :] = 0
        props = props / props.sum(dim=2, keepdim=True)  # normalize over all points of a given index tuple

        # weight the values by the proportions
        weights = mvalues[:, :, None, :].expand_as(props)
        # - add a dim for the MVNs

        weights = props * weights
        weights = weights.sum(dim=3) # - sum out the MVNs

        assert indices.size() == (b, 1, vs, 2), f'{indices.size()}, {(b, 1, vs, 2)}'
        assert weights.size() == (b, 1, vs), f'{weights.size()},  {(b, 1, vs)}'

        # expand for heads, fold heads into batch
        indices = indices[:, None, :, :, :].expand(b, h, 1, vs, 2).contiguous().view(b*h, vs, 2)
        weights = weights[:, None, :, :].expand(b, h, 1, vs).contiguous().view(b*h, vs)

        # compute keys, queries, values
        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)
        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4)) # b*h, t, e
        keys    = keys    / (e ** (1/4))

        # get dot product of queries and keys
        # - this will be a sparse matrix with the indices we've just computed, and values
        #   defined by the dot product

        # select the queries
        indflat = indices.view(b*h*vs, 2)
        ar = torch.arange(b*h, dtype=torch.long, device=d(x))[:, None].expand(b*h, vs).contiguous().view(b*h*vs)
        squeries = queries[ar, indflat[:, 0], :]
        skeys    = keys   [ar, indflat[:, 1], :]

        dot = torch.bmm(squeries[:, None, :], skeys[:, :, None]).view(b*h, vs)
        dot = sparse.logsoftmax(indices, weights * dot, s)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = sparse.batchmm(indices, dot, size=(t, t), xmatrix=values)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return self.unifyheads(out)


class ASHSelfAttention(nn.Module):
    """
    Masked sparse self attention. One degree of freedom, the receptive field is adaptive, based on the incoming
    embedding vector, position embedding and coordinate.
    """
    def __init__(self, emb, k, gadditional, radditional, region, heads=8, mask=False, min_sigma=0.05, sigma_scale=0.1, mmult = 1.0):
        """
        :param emb:
        :param k: Number of connections to the input for each output
        :param gadditional:
        :param radditional:
        :param region:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb, self.heads, self.mask, self.min_sigma, self.sigma_scale = emb, heads, mask, min_sigma, sigma_scale
        self.mmult = mmult

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

        self.gadditional = gadditional
        self.radditional = radditional
        self.region = region
        self.k = k

        self.register_buffer('mvalues', torch.ones((k, )))

        # network that generates the coordinates and sigmas
        hidden = emb * 4
        self.toparams = nn.Sequential(
            nn.Linear(emb + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, k * 3) # two means, one sigma
        )

    def hyper(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region

        # Generate coords
        coords = torch.arange(t, dtype=torch.float, device=d(x)) / t
        coords = coords[None, :, None,].expand(b, t, 1)

        input = torch.cat([x, coords], dim=2)
        params = self.toparams(input) # (b, t, k*3)

        assert not util.contains_nan(params),  f'params contain NaN\n intput {input.min()} {input.max()} \n {list(self.toparams.parameters())}'

        # Generate the logits that correspond to the diagonals of the matrix
        diags = torch.arange(t, dtype=torch.float, device=d(x))
        diags = util.inv(diags, mx=t)

        diags = diags[None, :, None, None].expand(b, t, k, 2)

        means =  params[:, :, :k*2].view(b, t, k, 2)
        sigmas = params[:, :, k*2:].view(b, t, k)
        values = self.mvalues[None, None, :].expand(b, t, k)

        means = diags + self.mmult * means
        means = util.flip(means)

        # means = util.flip(means.contiguous())  # flip everything to below the diagonal of the matrix

        s = (t, t)
        means, sigmas = sparse.transform_means(means, s), \
                        sparse.transform_sigmas(sigmas, s, min_sigma=self.min_sigma) * self.sigma_scale

        return means, sigmas, values

    def forward(self, x):

        b, t, e = x.size()
        h, k, reg = self.heads, self.k, self.region
        s = (t, t)

        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        means, sigmas, mvalues = self.hyper(x)

        # sample integer indices and values
        indices = sparse.ngenerate(means, self.gadditional, self.radditional, rng=(t, t), relative_range=(self.region, self.region), cuda=x.is_cuda)

        indices = util.flip(indices)

        indfl = indices.float()

        vs = k * (4 + self.radditional + self.gadditional)
        assert indices.size() == (b, t, vs, 2), f'{indices.size()}, {(b, t, vs, 2)}'

        # Mask for duplicate indices
        dups = util.nduplicates(indices).to(torch.bool)

        # compute (unnormalized) densities under the given MVNs (proportions)
        props = sparse.densities(indfl, means, sigmas).clone()
        props[dups, :] = 0
        props = props / props.sum(dim=2, keepdim=True)  # normalize over all points of a given index tuple

        # weight the values by the proportions
        weights = mvalues[:, :, None, :].expand_as(props)
        # - add a dim for the MVNs

        weights = props * weights
        weights = weights.sum(dim=3) # - sum out the MVNs

        assert indices.size() == (b, t, vs, 2), f'{indices.size()}, {(b, t, vs, 2)}'
        assert weights.size() == (b, t, vs), f'{weights.size()},  {(b, t, vs)}'

        # expand for heads, fold heads into batch
        indices = indices[:, None, :, :, :].expand(b, h, t, vs, 2).contiguous().view(b*h, t*vs, 2)
        weights = weights[:, None, :, :].expand(b, h, t, vs).contiguous().view(b*h, t*vs)

        # compute keys, queries, values
        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)
        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4)) # b*h, t, e
        keys    = keys    / (e ** (1/4))

        # get dot product of queries and keys
        # - this will be a sparse matrix with the indices we've just computed, and values
        #   defined by the dot product

        # select the queries
        indflat = indices.view(b*h*t*vs, 2)
        ar = torch.arange(b*h, dtype=torch.long, device=d(x))[:, None].expand(b*h, t*vs).contiguous().view(b*h*t*vs)
        squeries = queries[ar, indflat[:, 0], :]
        skeys    = keys   [ar, indflat[:, 1], :]

        dot = torch.bmm(squeries[:, None, :], skeys[:, :, None]).view(b*h,t*vs)

        assert not util.contains_nan(dot), f'dot contains nan (before softmax) {dot.min()}, {dot.mean()}, {dot.max()}'
        # print(f'before {dot.min()}, {dot.mean()}, {dot.max()}')

        dot = sparse.logsoftmax(indices, weights * dot, s).exp()
        # - dot now has row-wise self-attention probabilities
        # print(f'after  {dot.min()}, {dot.mean()}, {dot.max()}\n')

        assert not util.contains_nan(dot), f'dot contains nan (after softmax) {dot.min()}, {dot.mean()}, {dot.max()}'

        # apply the self attention to the values
        out = sparse.batchmm(indices, dot, size=(t, t), xmatrix=values)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        out = self.unifyheads(out)

        assert not util.contains_nan(out), f'output contains nan {out}'

        return out

class SelfAttention(nn.Module):
    """
    Plain, dense self attention
    """

    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys    = keys    / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

        assert not util.contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, sparse=False, **kwargs):
        super().__init__()

        if sparse:
            if mask:
                self.attention = ASHSelfAttention(emb, heads=heads, **kwargs)
            else:
                raise Exception('Not implemented yet')
        else:
            self.attention = SelfAttention(emb, heads=heads, mask=mask)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)
        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)
        x = self.do(x)

        return x

class GTransformer(nn.Module):
    """
    Transformer for generating text (character by character).
    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, sparse=False, **kwargs):
        """

        :param emb:
        :param heads:
        :param depth:
        :param seq_length:
        :param num_tokens:
        :param sparse:
        :param kwargs: Are passed to the sparse self attention
        """

        super().__init__()

        self.num_tokens = num_tokens
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=True, sparse=sparse, **kwargs))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_tokens)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d(x)))[None, :, :].expand(b, t, e)
        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)

    def forward_for_plot(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        means, sigmas, values = [], [], []

        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d(x)))[None, :, :].expand(b, t, e)
        x = tokens + positions

        for tblock in self.tblocks:
            m, s, v = tblock.attention.hyper(x)
            means.append(m)
            sigmas.append(s)
            values.append(v)

            x = tblock(x)

        return means, sigmas, values

def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    From https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py
    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    X = np.fromstring(open(path).read(n_train + n_valid + n_test), dtype=np.uint8)
    trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
    return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)

def go(arg):

    if arg.sparse:
        util.makedirs('./transformer-plots/')

    if arg.seed < 0:
        seed = random.randint(0, 1000000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    dv = 'cuda' if arg.cuda else 'cpu'

    tbw = SummaryWriter(log_dir=arg.tb_dir)

    # load the data
    data_train, data_val, data_test = enwik8(arg.data)
    data_test = data_test if arg.final else data_val

    # create the model
    if arg.sparse:
        model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context,
                             num_tokens=NUM_TOKENS, sparse=True, gadditional=arg.gadditional, radditional=arg.radditional,
                             region=arg.region, k=arg.k)
    else:
        model = GTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context, num_tokens=NUM_TOKENS)
    if arg.cuda:
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

    # training loop
    for i in tqdm.trange(arg.num_batches):

        if arg.lr_warmup > 0 and i < arg.lr_warmup:
            lr = max(  (arg.lr / arg.lr_warmup) * i, 1e-10)
            opt.lr = lr

        opt.zero_grad()

        # sample batches
        starts = torch.randint(size=(arg.batch_size, ), low=0, high=data_train.size(0) - arg.context - 1)
        seqs_source = [data_train[start  :start+arg.context  ] for start in starts]
        seqs_target = [data_train[start+1:start+arg.context+1] for start in starts]
        source = torch.cat([s[None, :] for s in seqs_source ], dim=0).to(torch.long)
        target = torch.cat([s[None, :] for s in seqs_target ], dim=0).to(torch.long)

        if arg.cuda:
            source, target = source.cuda(), target.cuda()

        source, target = Variable(source), Variable(target)

        output = model(source)

        loss = F.nll_loss(output.transpose(2, 1), target, reduction='none')

        # if i % 50 == 0:
        #     print(loss[0, 0], output[0, 0, input[0, 0]])
        #     sys.exit()

        loss = loss.mean()

        tbw.add_scalar('transformer/train-loss', float(loss.item()) * LOG2E, i * arg.batch_size)

        assert loss.item() == loss.item(), f'Loss is nan {loss}'

        loss.backward()

        assert not util.contains_nan(model.parameters()), f'Parameters have become NaN {model.parameters()}'

        # clip gradients
        if arg.gradient_clipping is not None:
            nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

        opt.step()

        if arg.sparse and arg.plot_every > 0 and i % arg.plot_every == 0:
            shape = (arg.context, arg.context)

            means, sigmas, values = model.forward_for_plot(source)
            for t, (m, s, v) in enumerate(zip(means, sigmas, values)):

                b, c, k, r = m.size()
                m = m.view(b, c*k, r)
                s = s.view(b, c*k, r)
                v = v.reshape(b, c*k)

                plt.figure(figsize=(7, 7))
                plt.cla()

                util.plot(m, s, v, shape=shape)
                plt.xlim((-MARGIN * (shape[0] - 1), (shape[0] - 1) * (1.0 + MARGIN)))
                plt.ylim((-MARGIN * (shape[0] - 1), (shape[0] - 1) * (1.0 + MARGIN)))

                plt.savefig(f'./transformer-plots/means{i:06}.{t}.pdf')

        if i != 0 and (i % arg.test_every == 0 or i == arg.num_batches - 1):

            upto = data_test.size(0) if i == arg.num_batches - 1 else arg.test_subset
            data_sub = data_test[:upto]

            with torch.no_grad():
                bits, tot = 0.0, 0
                batch = []

                for current in range(data_sub.size(0)):

                    fr = max(0, current - arg.context)
                    to = current + 1

                    context = data_sub[fr:to].to(torch.long)
                    if context.size(0) < arg.context + 1:
                        pad = torch.zeros(size=(arg.context + 1 - context.size(0),), dtype=torch.long)
                        context = torch.cat([pad, context], dim=0)

                        assert context.size(0) == arg.context + 1

                    if arg.cuda:
                        context = context.cuda()

                    batch.append(context[None, :])

                    if len(batch) == arg.test_batchsize or current == data_sub.size(0) - 1:

                        b = len(batch)

                        tot += b

                        all = torch.cat(batch, dim=0)
                        source = all[:, :-1]
                        target = all[:, -1]

                        output = model(source)

                        lnprobs = output[torch.arange(b, device=dv), -1, target]
                        log2probs = lnprobs * LOG2E

                        bits += - log2probs.sum()

                        batch = []

                assert tot == data_sub.size(0)

                bits_per_byte = bits / data_sub.size(0)

                print(f'epoch{i}: {bits_per_byte:.4} bits per byte')
                # print(f'epoch{i}: {bits:.4} total bits')

                tbw.add_scalar(f'transformer/eval-loss', bits_per_byte, i * arg.batch_size)

                # Generate from seed
                GENSIZE = 600
                TEMP = 0.5
                seedfr = random.randint(0, data_test.size(0) - arg.context)
                input = data_test[seedfr:seedfr + arg.context].to(torch.long)

                if arg.cuda:
                    input = input.cuda()

                input = Variable(input)

                print('[', end='', flush=True)
                for c in input:
                    print(str(chr(c)), end='', flush=True)
                print(']', end='', flush=True)

                for _ in range(GENSIZE):
                    output = model(input[None, :])
                    c = sample(output[0, -1, :], TEMP)
                    print(str(chr(max(32, c))), end='', flush=True)

                    input = torch.cat([input[1:], c[None]], dim=0)

                print()

if __name__ == "__main__":

    ## Parse the command line options
    parser = ArgumentParser()

    parser.add_argument("-N", "--num-batches",
                        dest="num_batches",
                        help="Number of batches to train on. Each batch contains randomly sampled subsequences of the data.",
                        default=1_000_000, type=int)

    parser.add_argument("-m", "--model",
                        dest="modelname",
                        help="Which model to train (dense, sparse).",
                        default='dense')

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=64, type=int)

    parser.add_argument("-k", "--num-points",
                        dest="k",
                        help="Number of index tuples per output in the sparse transformer.",
                        default=32, type=int)

    parser.add_argument("-a", "--gadditional",
                        dest="gadditional",
                        help="Number of additional points sampled globally",
                        default=8, type=int)

    parser.add_argument("-A", "--radditional",
                        dest="radditional",
                        help="Number of additional points sampled locally",
                        default=8, type=int)

    parser.add_argument("-R", "--region",
                        dest="region",
                        help="Size of the (square) region to use for local sampling.",
                        default=8, type=int)

    parser.add_argument("-c", "--cuda", dest="cuda",
                        help="Whether to use cuda.",
                        action="store_true")

    parser.add_argument("--sparse", dest="sparse",
                        help="Whether to use a sparse transformer.",
                        action="store_true")

    parser.add_argument("-D", "--data", dest="data",
                        help="Data file",
                        default=None)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-S", "--subsample",
                        dest="subsample",
                        help="Sample a subset of the indices to estimate gradients for",
                        default=None, type=float)

    parser.add_argument("-M", "--min-sigma",
                        dest="min_sigma",
                        help="Minimum value of sigma.",
                        default=0.0, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Data directory",
                        default=None)

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=70, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-C", "--context", dest="context",
                        help="Length of the sequences extracted from the corpus (and the context used during inference).",
                        default=300, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr of self-attention layers)",
                        default=4, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many batches between tests.",
                        default=1000, type=int)

    parser.add_argument("--plot-every",
                        dest="plot_every",
                        help="How many batches between plotting the sparse indices.",
                        default=100, type=int)

    parser.add_argument("--test-subset",
                        dest="test_subset",
                        help="A subset for the validation tests.",
                        default=100000, type=int)

    parser.add_argument("--test-batchsize",
                        dest="test_batchsize",
                        help="Batch size for computing the validation loss.",
                        default=1024, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=5000, type=int)

    options = parser.parse_args()

    print('OPTIONS ', options)

    go(options)
