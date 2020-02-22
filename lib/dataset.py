import json
import os

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


class ImageTipVelocitiesDataset(torch.utils.data.Dataset):
    def __init__(self, velocities_csv, metadata, root_dir, rotations_csv=None, transform=None,
                 initial_pixel_cropper=None, debug=False, get_rel_target_quantities=False):
        # convert to absolute path
        velocities_csv = os.path.abspath(velocities_csv)
        # if rotations csv is passed in, then dataset supplies vector of [tip velocities, rotations] (6 x 1)
        if rotations_csv is not None:
            rotations_csv = os.path.abspath(rotations_csv)
        metadata = os.path.abspath(metadata)
        root_dir = os.path.abspath(root_dir)

        self.tip_velocities_frame = pd.read_csv(velocities_csv, header=None)
        if rotations_csv is not None:
            self.rotations_csv_frame = pd.read_csv(rotations_csv, header=None)
            # Length must be the same
            assert len(self.tip_velocities_frame) == len(self.rotations_csv_frame)

        self.root_dir = root_dir
        self.transform = transform
        self.initial_pixel_cropper = initial_pixel_cropper
        self.debug = debug
        # Do we want target position and orientation, relative to robot?
        # This mode does not load images, for efficiency of training of baseline network
        self.get_rel_target_quantities = get_rel_target_quantities

        with open(metadata, "r") as m:
            metadata_content = m.read()
        self.demonstration_metadata = json.loads(metadata_content)

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

    def __len__(self):
        return len(self.tip_velocities_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if mode is relative quantities, do not load image for efficiency
        if not self.get_rel_target_quantities:
            img_name = os.path.join(self.root_dir, self.tip_velocities_frame.iloc[idx, 0])

            image = imageio.imread(img_name)
            if image.dtype == np.uint8:
                image = (image / 255).astype("float32")
        else:
            image = []

        tip_velocities = self.tip_velocities_frame.iloc[idx, 1:]
        tip_velocities = np.array(tip_velocities, dtype=np.float32)

        sample = {"image": image, "tip_velocities": tip_velocities}
        if self.rotations_csv_frame is not None:
            rotations = self.rotations_csv_frame.iloc[idx, 1:]
            rotations = np.array(rotations, dtype=np.float32)
            sample["rotations"] = rotations

        # get demonstration metadata
        # hack
        num_demonstration = self.tip_velocities_frame.iloc[idx, 0].split("image")[0]
        d_data = self.demonstration_metadata["demonstrations"][num_demonstration]
        instance_demonstration_idx = idx - d_data["start"]
        # if dataset is of type crop pixel, crop image using the metadata pixel, and add pixel information
        # if mode is relative quantities, we avoid this for efficiency
        if self.initial_pixel_cropper is not None and not self.get_rel_target_quantities:
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

        # add relative quantities
        if self.get_rel_target_quantities:
            sample["relative_target_position"] = np.array(
                d_data["relative_target_positions"][instance_demonstration_idx],
                dtype=np.float32
            )
            sample["relative_target_orientation"] = np.array(
                d_data["relative_target_orientations"][instance_demonstration_idx],
                dtype=np.float32
            )

        if self.debug and not self.get_rel_target_quantities:
            cv2.imshow("Image", sample["image"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Size image after crop:", sample["image"].shape)

        if self.transform and not self.get_rel_target_quantities:
            sample["image"] = self.transform(sample["image"])

        return sample
