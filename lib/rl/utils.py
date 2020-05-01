import enum

import numpy as np

from lib.common.test_utils import draw_crop, downsample_coordinates
from lib.common.utils import ResizeTransform, save_image
from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.cv.utils import CvUtils


class DatasetModality(enum.Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2


class CropScorer(object):
    def __init__(self, config):
        self.config = config
        self.resize = ResizeTransform(config["size"])
        width, height = config["size"]
        cropped_width, cropped_height = config["cropped_size"]
        divisor_width = int(width / cropped_width)
        divisor_height = int(height / cropped_height)
        self.dataset_get_crop_box = ImageTipVelocitiesDataset(
            velocities_csv=config["velocities_csv"],
            rotations_csv=config["rotations_csv"],
            metadata=config["metadata"],
            root_dir=config["root_dir"],
            transform=self.resize,
            ignore_cache_if_cropper=True,
            initial_pixel_cropper=TrainingPixelROI(
                480 // divisor_height, 640 // divisor_width
            )
        )

    def get_score(self, criterion, gt_demonstration_idx, width, height, predicted_center, cropped_width,
                  cropped_height):
        numpy_pixel_info = self.dataset_get_crop_box[gt_demonstration_idx]["pixel_info"]
        # Expected crop
        tl_gt = numpy_pixel_info["top_left"].astype(int)
        br_gt = numpy_pixel_info["bottom_right"].astype(int)
        tl_gt_down = downsample_coordinates(*tl_gt, og_width=640, og_height=480, to_width=width, to_height=height)
        br_gt_down = downsample_coordinates(*br_gt, og_width=640, og_height=480, to_width=width, to_height=height)

        # Predicted crop
        coords = CvUtils.get_bounding_box_coordinates(predicted_center[0], predicted_center[1],
                                                      cropped_height=cropped_height,
                                                      cropped_width=cropped_width)
        box = CvUtils.get_bounding_box(*coords)
        predicted_pixel_info_tl = box[0]
        predicted_pixel_info_br = box[3]

        score = criterion(tl_gt_down, predicted_pixel_info_tl, br_gt_down, predicted_pixel_info_br)
        return score, tl_gt, br_gt, predicted_pixel_info_tl, predicted_pixel_info_br


class CropTester(object):
    def __init__(self, config, test_env):
        self.config = config
        self.dataset_plain_images = ImageTipVelocitiesDataset(
            velocities_csv=config["velocities_csv"],
            rotations_csv=config["rotations_csv"],
            metadata=config["metadata"],
            root_dir=config["root_dir"]
        )
        self.resize = ResizeTransform(config["size"])
        width, height = config["size"]
        cropped_width, cropped_height = config["cropped_size"]
        divisor_width = int(width / cropped_width)
        divisor_height = int(height / cropped_height)
        self.dataset_get_crop_box = ImageTipVelocitiesDataset(
            velocities_csv=config["velocities_csv"],
            rotations_csv=config["rotations_csv"],
            metadata=config["metadata"],
            root_dir=config["root_dir"],
            transform=self.resize,
            ignore_cache_if_cropper=True,
            initial_pixel_cropper=TrainingPixelROI(
                480 // divisor_height, 640 // divisor_width
            )
        )
        self.scorer = CropScorer(config=config)
        # during training we expect reward to go up, but we also want to
        # use this to validate throughout training and to test the model
        self.env = test_env

    # for a single rollout
    def get_crop_score_per_rollout(self, criterion, model, save_images=False, log_dir="", prefix=""):
        if save_images and log_dir == "":
            raise ValueError("Where do we save the images?? -> log_dir")

        obs = self.env.reset()
        done = False
        count = 0
        scores = []
        width, height = self.config["size"]
        cropped_width, cropped_height = self.config["cropped_size"]

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            demonstration_index = self.env.get_curr_demonstration_idx()
            center = info["center_crop_pixel"]

            score, tl_gt, br_gt, predicted_pixel_info_tl, predicted_pixel_info_br = self.scorer.get_score(
                criterion=criterion,
                gt_demonstration_idx=demonstration_index,
                width=width,
                height=height,
                cropped_width=cropped_width,
                cropped_height=cropped_height,
                predicted_center=center
            )

            if save_images:
                numpy_image_plain_gt = self.dataset_plain_images[demonstration_index]
                image_gt = numpy_image_plain_gt["image"].copy()
                draw_crop(image_gt, tl_gt, br_gt, size=8)
                image_gt = self.resize(image_gt)
                draw_crop(image_gt, predicted_pixel_info_tl, predicted_pixel_info_br, red=True)
                save_image(image_gt, f"{log_dir}/{prefix}-{count}.png")

            scores.append(score)
            count += 1

        scores = np.array(scores)
        return np.mean(scores), np.std(scores)
