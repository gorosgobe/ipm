import json
import os

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset
import time


class TipVelocitiesDataset(torch.utils.data.Dataset):
    def __init__(self, velocities_csv, metadata, root_dir, rotations_csv=None):
        # convert to absolute path
        velocities_csv = os.path.abspath(velocities_csv)
        self.tip_velocities_frame = pd.read_csv(velocities_csv, header=None)
        if rotations_csv is not None:
            # if rotations csv is passed in, then dataset supplies vector of [tip velocities, rotations] (6 x 1)
            rotations_csv = os.path.abspath(rotations_csv)
            self.rotations_csv_frame = pd.read_csv(rotations_csv, header=None)
            # Length must be the same
            assert len(self.tip_velocities_frame) == len(self.rotations_csv_frame)

        self.root_dir = os.path.abspath(root_dir)
        metadata = os.path.abspath(metadata)
        with open(metadata, "r") as m:
            metadata_content = m.read()
        self.demonstration_metadata = json.loads(metadata_content)
        self.minimum_demonstration_length = self.compute_minimum_demonstration_length()

    def get_indices_for_demonstration(self, d_idx):
        demonstration_data = self.demonstration_metadata["demonstrations"][str(d_idx)]
        return demonstration_data["start"], demonstration_data["end"]

    def get_num_demonstrations(self):
        return self.demonstration_metadata["num_demonstrations"]

    def get_split(self, split_int, total_dems, start):
        n_split_demonstrations = int(split_int * total_dems)
        start_split, _ = self.get_indices_for_demonstration(start)
        _, end_split = self.get_indices_for_demonstration(start + n_split_demonstrations - 1)
        return Subset(self, np.arange(start_split, end_split + 1)), n_split_demonstrations

    def get_demonstration_metadata(self, idx):
        num_demonstration = self.tip_velocities_frame.iloc[idx, 0].split("image")[0]
        d_data = self.demonstration_metadata["demonstrations"][num_demonstration]
        return d_data

    def compute_minimum_demonstration_length(self):
        return min(self.demonstration_metadata["demonstrations"][d_str_idx]["num_tip_velocities"]
                   for d_str_idx in self.demonstration_metadata["demonstrations"])

    def get_minimum_demonstration_length(self):
        return self.minimum_demonstration_length

    def __len__(self):
        return len(self.tip_velocities_frame)


class BaselineTipVelocitiesDataset(TipVelocitiesDataset):
    def __init__(self, velocities_csv, metadata, root_dir, rotations_csv=None):
        super().__init__(velocities_csv=velocities_csv, metadata=metadata, root_dir=root_dir,
                         rotations_csv=rotations_csv)

    def __getitem__(self, idx):
        tip_velocities = self.tip_velocities_frame.iloc[idx, 1:]
        tip_velocities = np.array(tip_velocities, dtype=np.float32)

        sample = {"tip_velocities": tip_velocities}
        if self.rotations_csv_frame is not None:
            rotations = self.rotations_csv_frame.iloc[idx, 1:]
            rotations = np.array(rotations, dtype=np.float32)
            sample["rotations"] = rotations

        d_data = self.get_demonstration_metadata(idx)
        instance_demonstration_idx = idx - d_data["start"]

        # add relative quantities
        sample["relative_target_position"] = np.array(
            d_data["relative_target_positions"][instance_demonstration_idx],
            dtype=np.float32
        )
        sample["relative_target_orientation"] = np.array(
            d_data["relative_target_orientations"][instance_demonstration_idx],
            dtype=np.float32
        )

        return sample


class ImageTipVelocitiesDataset(TipVelocitiesDataset):
    def __init__(self, velocities_csv, metadata, root_dir, rotations_csv=None, transform=None,
                 initial_pixel_cropper=None, debug=False):
        super().__init__(velocities_csv=velocities_csv, metadata=metadata, root_dir=root_dir,
                         rotations_csv=rotations_csv)
        self.transform = transform
        self.initial_pixel_cropper = initial_pixel_cropper
        self.debug = debug
        self.cache = None
        self.initialising = True
        # Random crops around center pixel involves sampling, and we cannot save time by pre-computing it
        if self.initial_pixel_cropper is not None and not self.initial_pixel_cropper.is_random_crop():
            self.cache = {}
            print("Start cache loading...")
            start_time = time.time()
            for i in range(len(self)):
                self.cache[i] = self.__getitem__(i)
            print("Finished cache loading.")
            end_time = time.time()
            print(f"Elapsed seconds: {end_time - start_time}")
        self.initialising = False

    def __getitem__(self, idx):

        # to avoid continuously cropping, for simple, static simulation-based attention
        if self.initial_pixel_cropper is not None and not self.initialising and not self.initial_pixel_cropper.is_random_crop():
            return self.cache[idx]

        img_name = os.path.join(self.root_dir, self.tip_velocities_frame.iloc[idx, 0])

        image = imageio.imread(img_name)
        if image.dtype == np.uint8:
            image = (image / 255).astype("float32")

        tip_velocities = self.tip_velocities_frame.iloc[idx, 1:]
        tip_velocities = np.array(tip_velocities, dtype=np.float32)

        sample = {"image": image, "tip_velocities": tip_velocities}
        if self.rotations_csv_frame is not None:
            rotations = self.rotations_csv_frame.iloc[idx, 1:]
            rotations = np.array(rotations, dtype=np.float32)
            sample["rotations"] = rotations

        d_data = self.get_demonstration_metadata(idx)
        instance_demonstration_idx = idx - d_data["start"]

        # if dataset is of type crop pixel, crop image using the metadata pixel, and add pixel information
        # if mode is relative quantities, we avoid this for efficiency
        if self.initial_pixel_cropper is not None:
            pixels = d_data["crop_pixels"]
            pixel = pixels[instance_demonstration_idx]
            original_h, original_w, _channels = image.shape
            cropped_image, bounding_box_pixels = self.initial_pixel_cropper.crop(image, pixel)
            sample["image"] = cropped_image
            sample["pixel_info"] = {
                "top_left": np.array(bounding_box_pixels[1], dtype=np.float32),
                "bottom_right": np.array(bounding_box_pixels[4], dtype=np.float32),
                "original_image_height": np.array([original_h], dtype=np.float32),
                "original_image_width": np.array([original_w], dtype=np.float32)
            }

        if self.debug:
            cv2.imshow("Image", sample["image"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Size image after crop:", sample["image"].shape)

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample


