import torch
from torch.utils.data import Dataset
from torchvision import transforms

from lib.cv.dataset import ImagesOnlyDataset
from lib.dsae.dsae import CoordinateUtils


class DSAE_Dataset(Dataset):
    def __init__(self, velocities_csv, metadata, root_dir, reduced_transform, input_resize_transform, size,
                 add_coord=False):
        # dataset that loads three successor images
        # for idx t, returns t-1, t, t+1
        # if at the boundary of demonstration, return t=0, 1, 2 or t=n-2, n-1, n
        # so images for t=0 are same for t=1; images for t=n-1 are the same as t=n
        self.images_dataset = ImagesOnlyDataset(
            velocities_csv=velocities_csv,
            metadata=metadata,
            root_dir=root_dir,
            as_uint=True
        )
        self.input_resize_transform = input_resize_transform
        self.reduced_transform = reduced_transform
        self.add_coord = add_coord
        self.h, self.w = size

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

        # resize to input size
        prev_img = self.input_resize_transform(self.images_dataset[idx - 1])
        center_img = self.input_resize_transform(self.images_dataset[idx])
        next_img = self.input_resize_transform(self.images_dataset[idx + 1])
        imgs = (prev_img, center_img, next_img)

        # add coord maps if necessary
        normalising_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        if self.add_coord:
            image_x, image_y = CoordinateUtils.get_image_coordinates(self.h, self.w, normalise=False)
            normalising_transform = transforms.Compose([
                transforms.ToTensor(),  # toTensor
                transforms.Lambda(lambda x: torch.cat((x, image_x, image_y), dim=1)),
                # normalise with 5 channels
                transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5]),
            ])

        # get grayscaled output target image
        grayscaled = self.reduced_transform(imgs[center])  # resize to reduced size and grayscale
        sample = dict(
            images=torch.stack(list(map(lambda i: normalising_transform(i), imgs)), dim=0),
            center=center,
            target=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])(grayscaled)
        )
        return sample
