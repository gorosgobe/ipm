import torch
from torchvision import transforms

from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.common.test_utils import downsample_coordinates


class DSAE_Dataset(ImageTipVelocitiesDataset):
    def __init__(self, velocities_csv, rotations_csv, metadata, root_dir, reduced_transform, input_resize_transform,
                 size):
        # dataset that loads three successor images
        # if at boundary, load the boundary image twice
        super().__init__(
            velocities_csv=velocities_csv,
            rotations_csv=rotations_csv,
            metadata=metadata,
            root_dir=root_dir,
            as_uint=True
        )
        self.input_resize_transform = input_resize_transform
        self.reduced_transform = reduced_transform
        self.h, self.w = size

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        d_data = super().get_demonstration_metadata(idx)
        start_idx = d_data["start"]
        end_idx = d_data["end"]
        idx_prev = idx - 1
        idx_next = idx + 1

        if idx == start_idx:
            # idx, idx, idx + 1
            idx_prev = idx

        if idx == end_idx:
            # idx - 1, idx, idx
            idx_next = idx

        # resize to input size
        prev_img = self.input_resize_transform(super().__getitem__(idx_prev)["image"])

        curr_sample = super().__getitem__(idx)
        center_img = self.input_resize_transform(curr_sample["image"])
        center_target_vel_rot = torch.cat((
            torch.tensor(curr_sample["tip_velocities"]), torch.tensor(curr_sample["rotations"])
        ))
        next_img = self.input_resize_transform(super().__getitem__(idx_next)["image"])

        imgs = (prev_img, center_img, next_img)

        normalising_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # also add our expected crop pixel, for use when testing
        instance_demonstration_idx = idx - d_data["start"]
        pixels = d_data["crop_pixels"]
        pixel = pixels[instance_demonstration_idx]
        # pixel is in original 640x480 resolution, need to downsample
        pixel = downsample_coordinates(*pixel, og_height=480, og_width=640, to_height=self.h, to_width=self.w)

        # get grayscaled output target image
        grayscaled = self.reduced_transform(imgs[1])  # resize to reduced size and grayscale
        sample = dict(
            images=torch.stack(list(map(lambda i: normalising_transform(i), imgs)), dim=0),
            target_image=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])(grayscaled),
            target_vel_rot=center_target_vel_rot,
            pixel=torch.tensor(pixel)
        )
        return sample
