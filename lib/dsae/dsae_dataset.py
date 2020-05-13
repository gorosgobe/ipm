import time

import torch
from torchvision import transforms

from lib.cv.controller import TrainingPixelROI
from lib.common.test_utils import downsample_coordinates
from lib.cv.dataset import ImageTipVelocitiesDataset


class DSAE_Dataset(ImageTipVelocitiesDataset):
    def __init__(self, velocities_csv, rotations_csv, metadata, root_dir, input_resize_transform,
                 size, reduced_transform=transforms.Lambda(lambda x: x), single_image=False):
        # dataset that loads three successor images
        # if at boundary, load the boundary image twice
        # set single_image to true to only load the center image
        super().__init__(
            velocities_csv=velocities_csv,
            rotations_csv=rotations_csv,
            metadata=metadata,
            root_dir=root_dir,
            as_uint=True
        )
        self.input_resize_transform = input_resize_transform
        self.reduced_transform = reduced_transform
        # reduced transform can be left as identity if single_image is True
        self.single_image = single_image
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

        curr_sample = super().__getitem__(idx)
        center_img = self.input_resize_transform(curr_sample["image"])
        center_target_vel_rot = torch.cat((
            torch.tensor(curr_sample["tip_velocities"]), torch.tensor(curr_sample["rotations"])
        ))

        if not self.single_image:
            # resize to input size
            prev_img = self.input_resize_transform(super().__getitem__(idx_prev)["image"])
            next_img = self.input_resize_transform(super().__getitem__(idx_next)["image"])
            # get grayscaled output target image
            imgs = (prev_img, center_img, next_img)
            grayscaled = self.reduced_transform(imgs[1])  # resize to reduced size and grayscale
        else:
            imgs = [center_img]
            # get grayscaled output target image
            grayscaled = self.reduced_transform(imgs[0])  # resize to reduced size and grayscale

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

        sample = dict(
            images=torch.stack(list(map(lambda i: normalising_transform(i), imgs)), dim=0),
            target_image=(transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])(grayscaled) if not self.single_image else None),
            target_vel_rot=center_target_vel_rot,
            pixel=torch.tensor(pixel)
        )
        return sample


class DSAE_FeatureProviderDataset(DSAE_Dataset):
    def __init__(self, feature_provider, velocities_csv, rotations_csv, metadata, root_dir,
                 input_resize_transform, size, cache, reduced_transform=None, add_pixel=False, add_image=False):
        super().__init__(velocities_csv, rotations_csv, metadata, root_dir, input_resize_transform,
                         size, single_image=True)
        self.feature_provider = feature_provider
        self.add_pixel = add_pixel
        self.add_image = add_image
        self.cache = cache
        self.cache_content = {}
        self.initialising = False
        if self.cache:
            print("Loading feature cache...")
            before = time.time()
            self.initialising = True
            for i in range(len(self)):
                self.cache_content[i] = self.__getitem__(i)
            self.initialising = False
            print(f"Loading finished, {time.time() - before} seconds elapsed")

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if not self.initialising and self.cache:
            return self.cache_content[idx]

        sample = super().__getitem__(idx)
        # features of size (latent // 2, 2), convert to (latent,)
        features = self.feature_provider(sample["images"][0]).squeeze(0).view(-1)
        # move to cpu here to avoid multiprocessing errors when accessing on cache
        feature_sample = dict(
            features=features.cpu(),
            target_vel_rot=sample["target_vel_rot"].cpu(),
            **(dict(image=sample["images"][0].cpu()) if self.add_image else {}),
            **(dict(pixel=sample["pixel"].cpu()) if self.add_pixel else {}),
        )
        return feature_sample


class DSAE_SingleFeatureProviderDataset(object):
    def __init__(self, feature_provider_dataset, feature_index):
        self.feature_provider_dataset = feature_provider_dataset
        self.feature_index = feature_index

    def __len__(self):
        return self.feature_provider_dataset.__len__()

    def __getitem__(self, idx):
        # feature sample is in CPU
        feature_sample = self.feature_provider_dataset[idx]
        assert "image" in feature_sample
        assert "features" in feature_sample
        assert "target_vel_rot" in feature_sample

        single_feature = feature_sample["features"].view(-1, 2)[self.feature_index]
        sample = dict(
            feature=single_feature,
            image=feature_sample["image"],
            target_vel_rot=feature_sample["target_vel_rot"]
        )
        return sample


class DSAE_FeatureCropTVEAdapter(object):
    def __init__(self, single_feature_dataset, crop_size, size=(128, 96), add_spatial_maps=True):
        self.single_feature_dataset = single_feature_dataset
        self.w, self.h = size
        cropped_width, cropped_height = crop_size
        self.pixel_cropper = TrainingPixelROI(
            cropped_height=cropped_height,
            cropped_width=cropped_width,
            add_spatial_maps=add_spatial_maps
        )
        self.normalising_transform = transforms.Compose([
            transforms.ToTensor(),
            # image is already normalised
            transforms.Normalize([0, 0, 0, 0.5, 0.5], [1, 1, 1, 0.5, 0.5])
        ])

    def __len__(self):
        return self.single_feature_dataset.__len__()

    def __getitem__(self, idx):
        # sample is in CPU
        sample = self.single_feature_dataset[idx]
        tip_velocities, rotations = torch.split(sample["target_vel_rot"], 3)
        feature = (sample["feature"] + 1) / 2
        # scale pixel to original size, such as 128, 96 range
        pixel = (feature * torch.tensor([self.w - 1, self.h - 1], dtype=torch.float32)).type(dtype=torch.int32)
        cropped_image, _ = self.pixel_cropper.crop(sample["image"].numpy().transpose(1, 2, 0), pixel)
        return dict(
            image=self.normalising_transform(cropped_image),
            tip_velocities=tip_velocities,
            rotations=rotations
        )
