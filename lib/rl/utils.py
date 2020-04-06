import enum

import numpy as np

from demonstration_env import SingleDemonstrationEnv
from lib.common.test_utils import draw_crop, downsample_coordinates
from lib.common.utils import ResizeTransform, save_image
from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.cv.utils import CvUtils


class CropTestModality(enum.Enum):
    TRAINING = 0
    VALIDATION = 1,
    TEST = 2


class CropTester(object):
    def __init__(self, config, demonstration_dataset, crop_test_modality, environment_klass=SingleDemonstrationEnv):
        self.config = config
        self.dataset_plain_images = ImageTipVelocitiesDataset(
            velocities_csv=config["velocities_csv"],
            rotations_csv=config["rotations_csv"],
            metadata=config["metadata"],
            root_dir=config["root_dir"]
        )
        self.resize = ResizeTransform(config["size"])
        self.dataset_get_crop_box = ImageTipVelocitiesDataset(
            velocities_csv=config["velocities_csv"],
            rotations_csv=config["rotations_csv"],
            metadata=config["metadata"],
            root_dir=config["root_dir"],
            transform=self.resize,
            # TODO: automate the // 2 depending on size and cropped size
            initial_pixel_cropper=TrainingPixelROI(
                480 // 2, 640 // 2, add_spatial_maps=False
            ),
            force_not_cache=True
        )
        # during training we expect reward to go up, but we also want to
        # use this to validate throughout training and to test the model
        self.env = environment_klass(
            demonstration_dataset=demonstration_dataset,
            config=config,
            use_split_idx=crop_test_modality,
            skip_reward=True
        )

    # for a single rollout
    def get_crop_score(self, criterion, model, save_images=False, log_dir="", prefix=""):
        if save_images and log_dir == "":
            raise ValueError("Where do we save the images?? -> log_dir")

        obs = self.env.reset()
        done = False
        count = 0
        scores = []
        while not done:
            demonstration_index = self.env.demonstration_img_idx
            numpy_image_plain_gt = self.dataset_plain_images[demonstration_index]
            numpy_pixel_info = self.dataset_get_crop_box[demonstration_index]["pixel_info"]
            image_gt = numpy_image_plain_gt["image"].copy()
            tl_gt = numpy_pixel_info["top_left"].astype(int)
            br_gt = numpy_pixel_info["bottom_right"].astype(int)
            if save_images:
                draw_crop(image_gt, tl_gt, br_gt, size=4)
            image_gt = self.resize(image_gt)

            action, _states = model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            center = info["center_crop_pixel"]
            cropped_width, cropped_height = self.config["cropped_size"]
            coords = CvUtils.get_bounding_box_coordinates(center[0], center[1], cropped_height=cropped_height,
                                                          cropped_width=cropped_width)
            box = CvUtils.get_bounding_box(*coords)
            predicted_pixel_info_tl = box[0]
            predicted_pixel_info_br = box[3]
            if save_images:
                draw_crop(image_gt, predicted_pixel_info_tl, predicted_pixel_info_br, red=True)
                save_image(image_gt, f"{log_dir}/{prefix}-{count}.png")

            width, height = self.config["size"]
            tl_gt_down = downsample_coordinates(*tl_gt, og_width=640, og_height=480, to_width=width, to_height=height)
            br_gt_down = downsample_coordinates(*br_gt, og_width=640, og_height=480, to_width=width, to_height=height)
            score = criterion(tl_gt_down, predicted_pixel_info_tl, br_gt_down, predicted_pixel_info_br)
            scores.append(score)
            count += 1

        scores = np.array(scores)
        return np.mean(scores), np.std(scores)
