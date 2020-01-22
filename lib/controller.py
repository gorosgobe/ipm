import time

import cv2
import numpy as np
import torch

from lib import utils
from lib.camera import Camera


class IdentityCropper(object):
    def crop(self, image):
        return image


class OffsetCropper(object):
    def __init__(self, cropped_height, cropped_width, offset_height=0, offset_width=0):
        self.cropped_height = cropped_height
        self.cropped_width = cropped_width
        self.offset_height = offset_height
        self.offset_width = offset_width

    def crop(self, image):
        height, width, _ = image.shape
        center_x = width // 2 + self.offset_width
        center_y = height // 2 + self.offset_height
        half_size_height = self.cropped_height // 2
        half_size_width = self.cropped_width // 2
        cropped_image = image[
                        center_y - half_size_height:center_y + half_size_height,
                        center_x - half_size_width:center_x + half_size_width
                        ]
        return cropped_image


class TruePixelROI(object):
    def __init__(self, cropped_height, cropped_width, pixel_position_estimator, target_object):
        """
        :param cropped_height: Height of region to crop
        :param cropped_width: Width of region to crop
        :param pixel_position_estimator: object that, given a handle, can compute its screen, pixel position for
        full resolution of image supplied to "crop"
        :param target_object: handle of target object, to compute pixel position of
        """
        self.cropped_height = cropped_height
        self.cropped_width = cropped_width
        self.target_object = target_object
        self.pixel_position_estimator = pixel_position_estimator

    def crop(self, image):
        height, width, _ = image.shape
        pixel, _ = self.pixel_position_estimator.compute_pixel_position(self.target_object.get_handle())
        center_x, center_y = pixel
        half_size_height = self.cropped_height // 2
        half_size_width = self.cropped_width // 2

        dx = 0
        dy = 0
        if center_x + half_size_width >= width:
            dx = -(center_x + half_size_width - width) - 1
        elif center_x - half_size_width < 0:
            dx = -(center_x - half_size_width)

        if center_y + half_size_height >= height:
            dy = -(center_y + half_size_height - height) - 1
        elif center_y - half_size_height < 0:
            dy = -(center_y - half_size_height)

        # otherwise, crop lies fully inside the image, dx, dy = 0 apply

        center_x += dx
        center_y += dy

        cropped_image = image[
                        center_y - half_size_height:center_y + half_size_height + 1,
                        center_x - half_size_width:center_x + half_size_width + 1
                        ]

        return cropped_image


# Pixel ROI for training, used to determine what the model sees at training time
# Pixel is loaded from data, so wrap for interfacing with TruePixelROI
class LoadedPixelEstimator(object):
    def __init__(self, pixel):
        self.pixel = pixel

    def compute_pixel_position(self, handle):
        return self.pixel, None


# To pass in by TrainingPixelROI to TruePixelROI for pixel estimators that return pixel from data,
# such as LoadedPixelEstimator
class FakeHandle(object):
    def get_handle(self):
        return None


class TrainingPixelROI(object):
    """
    Region of interest for training, where pixel estimator is just the pixel stored for the image
    """

    def __init__(self, cropped_height, cropped_width):
        self.cropped_height = cropped_height
        self.cropped_width = cropped_width

    def crop(self, image, pixel):
        loaded_pixel_estimator = LoadedPixelEstimator(pixel)
        true_pixel_roi = TruePixelROI(self.cropped_height, self.cropped_width, loaded_pixel_estimator, FakeHandle())
        return true_pixel_roi.crop(image)


# Controller that applies transformations and ROI to image at test time, and estimates tip velocity
class TipVelocityController(object):
    def __init__(self, tve_model, roi_estimator, debug=False):
        self.tip_velocity_estimator = tve_model
        self.roi_estimator = roi_estimator
        self.debug = debug

    def get_model(self):
        return self.tip_velocity_estimator

    def get_tip_control(self, image):
        # select region of interest (manual crop or RL agent)
        image = self.roi_estimator.crop(image)
        # resizes image
        image = self.tip_velocity_estimator.resize_image(image)
        # apply normalisation and other transforms as required
        transformed_image = self.tip_velocity_estimator.transforms(image)

        with torch.no_grad():
            image_tensor = torch.unsqueeze(transformed_image, 0)
            # batch with single tip control command
            tip_control_single_batch = self.tip_velocity_estimator.predict(image_tensor)
            tip_control = tip_control_single_batch[0]

        return tip_control
