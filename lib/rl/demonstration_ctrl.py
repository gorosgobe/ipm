from functools import reduce

import numpy as np


class DemonstrationSampler(object):
    def __init__(self, split, num_demonstrations, dataset_type_idx, random_provider=np.random.choice):
        self.split = split
        self.dataset_type_idx = dataset_type_idx
        self.num_demonstrations = num_demonstrations
        self.random_provider = random_provider

    def sample_demonstration(self, size):
        num_dems = int(self.split[self.dataset_type_idx] * self.num_demonstrations)
        demonstration_idxs = self.random_provider(num_dems,
                                                  size=size, replace=False)
        demonstration_idxs = list(map(self.get_global_demonstration_index, demonstration_idxs))
        return demonstration_idxs

    def get_demonstration_indexer(self, demonstration_dataset, demonstrations):
        # sample demonstrations
        demonstration_idxs = self.sample_demonstration(size=demonstrations)
        # training demonstration
        pair_list = list(map(demonstration_dataset.get_indices_for_demonstration, demonstration_idxs))
        return DemonstrationIndexer(
            *pair_list,
            demonstration_dataset=demonstration_dataset
        )

    def get_global_demonstration_index(self, index):
        return int(
            sum(self.split[:self.dataset_type_idx]) * self.num_demonstrations
        ) + index


class DemonstrationIndexer(object):
    def __init__(self, *start_end_pairs, demonstration_dataset):
        self.demonstration_dataset = demonstration_dataset
        self.start_end_pairs = start_end_pairs
        self.lengths = list(map(lambda pair: pair[1] - pair[0] + 1, start_end_pairs))
        self.indices = reduce(lambda x, y: x + y, map(lambda pair: list(range(pair[0], pair[1] + 1)), start_end_pairs))
        self.curr_idx = 0
        self.length_idx = 0
        self.length_count = 0

    def advance(self):
        self.curr_idx += 1
        self.length_count += 1
        # if past a demonstration, advance the length index too, used to determine which demonstration we are in
        if self.lengths[self.length_idx] == self.length_count:
            self.length_count = 0
            self.length_idx += 1

    def get_which_demonstration(self):
        # returns the index of the current demonstration (not of the current image), wrt the ones passed in
        return self.length_idx

    def get_curr_demonstration_data(self):
        return self.demonstration_dataset[self.get_curr_demonstration_idx()]

    def get_prev_demonstration_data(self):
        return self.demonstration_dataset[self.get_prev_demonstration_idx()]

    def get_idx(self, idx):
        assert idx in range(len(self.indices))
        return self.indices[idx]

    def get_curr_demonstration_idx(self):
        return self.get_idx(self.curr_idx)

    def get_prev_demonstration_idx(self):
        return self.get_idx(self.curr_idx - 1)

    def done(self):
        return self.curr_idx == len(self.indices)

    def get_length(self):
        return len(self.indices)

    def get_lengths(self):
        return self.lengths
