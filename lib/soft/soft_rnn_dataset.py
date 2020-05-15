import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
from torch.utils.data._utils.collate import default_collate

from dsae.dsae import CoordinateUtils
from lib.cv.dataset import ImageTipVelocitiesDataset


class SoftRNNDataset(ImageTipVelocitiesDataset):
    def __init__(self, velocities_csv, metadata, root_dir, rotations_csv, cache, transform, is_coord):
        super().__init__(
            velocities_csv=velocities_csv,
            metadata=metadata,
            root_dir=root_dir,
            rotations_csv=rotations_csv,
            transform=transform
        )
        self.is_coord = is_coord
        self.cache = cache
        if self.cache:
            self.initialising = True
            self.cache_content = {}
            before = time.time()
            print("Loading cache...")
            for i in range(len(self)):
                self.cache_content[i] = self.__getitem__(i)
            print(f"Finished loading, {time.time() - before}s elapsed")
        self.initialising = False

    @staticmethod
    def custom_collate(batch):
        demonstration = [default_collate(data_dict["demonstration"]) for data_dict in batch]
        lengths = default_collate([len(data_dict["demonstration"]) for data_dict in batch])
        padded_demonstration = pad_sequence(demonstration, batch_first=True)
        targets = [default_collate(data_dict["demonstration_targets"]) for data_dict in batch]
        padded_targets = pad_sequence(targets, batch_first=True)
        return dict(
            demonstration=padded_demonstration,
            demonstration_targets=padded_targets,
            lengths=lengths
        )

    def get_split(self, split_int, total_dems, start):
        n_split_demonstrations = int(split_int * total_dems)
        return Subset(self, np.arange(start, start + n_split_demonstrations)), n_split_demonstrations

    def __len__(self):
        return self.get_num_demonstrations()

    def __getitem__(self, idx):
        if self.cache and not self.initialising:
            return self.cache_content[idx]
        # idx is for demonstration, get images associated with this demonstration
        start, end = self.get_indices_for_demonstration(idx)
        demonstration_data = [super(SoftRNNDataset, self).__getitem__(i) for i in range(start, end + 1)]
        demonstration_images = [dem_data["image"] for dem_data in demonstration_data]
        demonstration_targets = [
            np.concatenate((dem_data["tip_velocities"], dem_data["rotations"])) for dem_data in demonstration_data
        ]

        if self.is_coord:
            result = []
            c, h, w = demonstration_images[0].size()
            image_x, image_y = CoordinateUtils.get_image_coordinates(h, w, normalise=True)
            image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
            image_coordinates = image_coordinates.permute(2, 0, 1)
            # (2, H, W)
            for image in demonstration_images:
                res = torch.cat((image, image_coordinates), dim=0)
                assert res.size() == (5, h, w)
                result.append(res)
            demonstration_images = result
        return dict(
            demonstration=demonstration_images,
            demonstration_targets=demonstration_targets,
        )
