import json
import os

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset


class ImageTipVelocitiesDataset(torch.utils.data.Dataset):
    def __init__(self, csv, metadata, root_dir, transform=None, cache_images=True, initial_pixel_cropper=None, debug=False):
        # convert to absolute path
        csv = os.path.abspath(csv)
        metadata = os.path.abspath(metadata)
        root_dir = os.path.abspath(root_dir)

        self.tip_velocities_frame = pd.read_csv(csv, header=None)
        self.root_dir = root_dir
        self.transform = transform
        self.initial_pixel_cropper = initial_pixel_cropper
        self.debug = debug

        with open(metadata, "r") as m:
            metadata_content = m.read()
        self.demonstration_metadata = json.loads(metadata_content)
        self.cache_images = cache_images
        self.cache = {}
        if self.cache_images:
            print("Loading images into memory...")
            # hack to preload all images from cache
            for i in range(len(self)):
                self.__getitem__(i)
                print("Loaded ", i)
            print("Finished loading.")

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

        img_name = os.path.join(self.root_dir, self.tip_velocities_frame.iloc[idx, 0])

        if not self.cache_images:
            image = imageio.imread(img_name)
        else:
            if img_name not in self.cache:
                self.cache[img_name] = imageio.imread(img_name)
            image = self.cache[img_name]

        tip_velocities = self.tip_velocities_frame.iloc[idx, 1:]
        tip_velocities = np.array(tip_velocities, dtype=np.float32)

        sample = {'image': image, 'tip_velocities': tip_velocities}

        # if dataset is of type crop pixel, crop image using the metadata pixel
        if self.initial_pixel_cropper is not None:
            # hack
            num_demonstration = self.tip_velocities_frame.iloc[idx, 0].split("image")[0]
            d_data = self.demonstration_metadata["demonstrations"][num_demonstration]
            pixels = d_data["crop_pixels"]
            if self.debug:
                print("Should be #", idx - d_data["start"], "pixel, from", img_name)
            pixel = pixels[idx - d_data["start"]]
            sample["image"] = self.initial_pixel_cropper.crop(image, pixel)

        if self.debug:
            cv2.imshow("Image", sample["image"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("Size image after crop:", sample["image"].shape)

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample