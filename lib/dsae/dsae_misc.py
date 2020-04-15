import torch
from torch import nn
from torch.utils.data import Dataset

from lib.cv.dataset import ImagesOnlyDataset


class DSAE_Dataset(Dataset):
    def __init__(self, velocities_csv, metadata, root_dir, reduced_transform, transform=None):
        # dataset that loads three successor images
        # for idx t, returns t-1, t, t+1
        # if at the boundary of demonstration, return t=0, 1, 2 or t=n-2, n-1, n
        # so images for t=0 are same for t=1; images for t=n-1 are the same as t=n
        self.images_dataset = ImagesOnlyDataset(
            velocities_csv=velocities_csv,
            metadata=metadata,
            root_dir=root_dir,
            transform=transform
        )
        self.reduced_transform = reduced_transform

    def __len__(self):
        return self.images_dataset.__len__()

    def __getitem__(self, idx):
        d_data = self.images_dataset.get_demonstration_metadata(idx)
        start_idx = d_data["start"]
        end_idx = d_data["end"]
        center = 1

        if idx == start_idx:
            # idx, idx + 1, idx + 2
            idx += 1
            center = 0

        if idx == end_idx:
            # idx _2, idx - 1, idx
            idx -= 1
            center = 2

        prev_img = self.images_dataset[idx - 1]
        center_img = self.images_dataset[idx]
        next_img = self.images_dataset[idx + 1]

        sample = dict(images=torch.stack((prev_img, center_img, next_img), dim=0), center=center)
        # get grayscaled output target image
        grayscaled = self.reduced_transform(sample["images"][center])
        return dict(**sample, target=grayscaled)


class DSAE_Loss(object):
    def __init__(self, add_g_slow=True):
        self.add_g_slow = add_g_slow
        self.mse_loss = nn.MSELoss()

    def __call__(self, reconstructed, target, ft_minus1, ft, ft_plus1):
        loss = self.mse_loss(reconstructed, target)
        if self.add_g_slow:
            loss += self.mse_loss(ft_plus1 - ft, ft - ft_minus1)
        return loss
