# Copyright (C) Alibaba Group Holding Limited. 

import torch
import torch.distributed as dist
import math
import itertools
from torch.utils.data.sampler import Sampler


__all__ = ['InfiniteSampler']

class InfiniteSampler(Sampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=None,
                 batch_size=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if seed is None:
            seed = 8888
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size

        # compute total size
        batch_size = 1 if batch_size is None else batch_size
        divisor = self.num_replicas * batch_size
        self.total_size = int(math.ceil(len(self.dataset) / float(divisor)) * divisor)
        self.num_samples = self.total_size // self.num_replicas
    
    def __iter__(self):
        start, stop, step = self.rank, None, self.num_replicas
        yield from itertools.islice(
            self._infinite_indices(), start, stop, step)
    
    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
            
            padded_indices = []
            for rank in range(self.num_replicas):
                rank_indices = indices[rank::self.num_replicas]
                rank_indices += rank_indices[:(self.num_samples - len(rank_indices))]
                padded_indices.append(rank_indices)
            padded_indices = list(itertools.chain.from_iterable(zip(*padded_indices)))
            assert len(padded_indices) == self.total_size

            yield from padded_indices
    
    @property
    def length(self):
        return self.num_samples
