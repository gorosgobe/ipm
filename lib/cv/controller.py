import abc
import enum
from abc import ABC

import numpy as np
import torch
import torchvision

from lib.common import utils
from lib.cv.utils import CvUtils


class IdentityCropper(object):
    def crop(self, image):
        return image, None


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


class CropDeviationSampler(object):
    def __init__(self, std):
        self.std = std

    def sample(self):
        return np.random.normal(loc=0.0, scale=self.std, size=2).astype(int)


class SpatialDimensionAdder(object):
    def __init__(self):
        self.device = utils.set_up_cuda(-1, False)
        self.to_tensor = torchvision.transforms.ToTensor()

    def get_tensor_batch_spatial_dimensions(self, b, h, w):
        i, j = self.get_spatial_dimensions(h, w)
        i_tensor = self.to_tensor(i).to(self.device).unsqueeze(0).expand((b, 1, h, w))
        j_tensor = self.to_tensor(j).to(self.device).unsqueeze(0).expand((b, 1, h, w))
        return i_tensor, j_tensor

    @staticmethod
    def get_spatial_dimensions(height, width):
        i = np.tile(np.array(range(width), dtype=np.float32), (height, 1))
        i = np.expand_dims(i, axis=2) / (width - 1)
        j = np.tile(np.array(range(height), dtype=np.float32), (width, 1)).T
        j = np.expand_dims(j, axis=2) / (height - 1)
        return i, j

    @staticmethod
    def add_spatial_dimensions(image):
        # Add spatial map and normalise
        height, width, _channels = image.shape
        i, j = SpatialDimensionAdder.get_spatial_dimensions(height, width)
        return np.concatenate((image, i, j), axis=2)


class ROI(abc.ABC):
    def is_random_crop(self):
        raise NotImplementedError("is_random_crop is not implemented")


class TrainingROI(ROI, ABC):
    def crop(self, image, pixel):
        raise NotImplementedError("crop is not implemented")


class TruePixelROI(ROI):
    def __init__(self, cropped_height, cropped_width, pixel_position_estimator, target_object, add_spatial_maps=False,
                 crop_deviation_sampler=None):
        """
        :param cropped_height: Height of region to crop
        :param cropped_width: Width of region to crop
        :param pixel_position_estimator: object that, given a handle, can compute its screen, pixel position for
        full resolution of image supplied to "crop"
        :param target_object: handle of target object, to compute pixel position of
        :param add_spatial_maps: Add a spatial feature map to the cropped image, as in
        "An intriguing failing of convolutional neural networks and the CoordConv solution",
        https://arxiv.org/pdf/1807.03247.pdf
        :param crop_deviation_sampler: If set to None (default), image is cropped at provided/computed pixel. Otherwise,
        an offset is sampled (.sample()) from the provided object, added to the computed pixel, and then the image is cropped.
        Note that if the sampled offset makes the crop lie outside of the image, the crop is moved into the image (normal behaviour).
        """
        self.cropped_height = cropped_height
        self.cropped_width = cropped_width
        self.target_object = target_object
        self.pixel_position_estimator = pixel_position_estimator
        self.add_spatial_maps = add_spatial_maps
        self.crop_deviation_sampler = crop_deviation_sampler

    def is_random_crop(self):
        return self.crop_deviation_sampler is not None

    def crop(self, image):
        if self.add_spatial_maps:
            image = SpatialDimensionAdder.add_spatial_dimensions(image)

        height, width, _ = image.shape
        pixel, _ = self.pixel_position_estimator.compute_pixel_position(self.target_object.get_handle())
        center_x, center_y = pixel
        if self.crop_deviation_sampler is not None:
            offset = self.crop_deviation_sampler.sample()
            center_x += offset[0]
            center_y += offset[1]

        center_x, center_y = CvUtils.fit_crop_to_image(center_x, center_y, height, width, self.cropped_height,
                                                       self.cropped_width)

        x_min, x_max, y_min, y_max = CvUtils.get_bounding_box_coordinates(center_x, center_y, self.cropped_height,
                                                                          self.cropped_width)
        if x_min < 0 or x_max < 0 or y_min < 0 or y_max < 0:
            raise ValueError("LESS THAN ZERO")
        # center, top left, top right, bottom left, bottom right
        bounding_box_pixels = [
            (center_x, center_y), *CvUtils.get_bounding_box(x_min, x_max, y_min, y_max)
        ]

        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        return cropped_image, bounding_box_pixels


# Pixel ROI for training, used to determine what the model sees at training time

# Pixel is loaded from data, so wrap for interfacing with TruePixelROI
class LoadedPixelEstimator(object):
    def __init__(self, pixel):
        self.pixel = pixel

    def compute_pixel_position(self, _handle):
        return self.pixel, None


# To pass in by TrainingPixelROI to TruePixelROI for pixel estimators that return pixel from data,
# such as LoadedPixelEstimator
class FakeHandle(object):
    def get_handle(self):
        return None


class TrainingPixelROI(TrainingROI):
    """
    Region of interest for training, where estimated pixel is just the pixel stored for the image.
    """

    def __init__(self, cropped_height, cropped_width, add_spatial_maps=False, crop_deviation_sampler=None):
        self.cropped_height = cropped_height
        self.cropped_width = cropped_width
        self.add_spatial_maps = add_spatial_maps
        self.crop_deviation_sampler = crop_deviation_sampler

    def is_random_crop(self):
        return self.crop_deviation_sampler is not None

    def crop(self, image, pixel):
        loaded_pixel_estimator = LoadedPixelEstimator(pixel)
        true_pixel_roi = TruePixelROI(
            self.cropped_height,
            self.cropped_width,
            loaded_pixel_estimator,
            FakeHandle(),
            self.add_spatial_maps,
            self.crop_deviation_sampler
        )
        return true_pixel_roi.crop(image)


class RandomPixelROI(TrainingPixelROI):
    """
    Region of interest for meta-training, where estimated pixel is just a random pixel in the image
    """
    def __init__(self, cropped_height, cropped_width, add_spatial_maps=False, random_provider=np.random.choice):
        super().__init__(cropped_height, cropped_width, add_spatial_maps)
        self.random_provider = random_provider

    def is_random_crop(self):
        return True

    def crop(self, image, _loaded_pixel):
        # ignore loaded pixel, get a random one from the image
        height, width, _ = image.shape
        random_pixel = self.get_random_pixel(height, width)
        return super().crop(image, random_pixel)

    def get_random_pixel(self, height, width):
        # size of possible pixels can be computed with convolution formula w/o padding
        range_x = width - self.cropped_width + 1
        range_y = height - self.cropped_height + 1
        start_x = (self.cropped_width // 2) - (self.cropped_width % 2 == 0)
        start_y = (self.cropped_height // 2) - (self.cropped_height % 2 == 0)
        dx = self.random_provider(np.arange(range_x))
        dy = self.random_provider(np.arange(range_y))
        return start_x + dx, start_y + dy


class ControllerType(enum.Enum):
    DEFAULT = 0
    TOP_LEFT_BOTTOM_RIGHT_PIXELS = 2
    RELATIVE_POSITION_AND_ORIENTATION = 3


# Controller that applies transformations and ROI to image at test time, and estimates tip velocity
class TipVelocityController(object):
    def __init__(self, tve_model, roi_estimator, target_object=None, camera=None,
                 controller_type=ControllerType.DEFAULT, debug=False):
        self.tip_velocity_estimator = tve_model
        self.roi_estimator = roi_estimator
        self.target_object = target_object
        self.camera = camera
        if (
                self.target_object is None or self.camera is None) and controller_type == ControllerType.RELATIVE_POSITION_AND_ORIENTATION:
            raise ValueError(
                "Controller requires target object and camera handle to compute relative positions and orientations")
        self.debug = debug
        self.controller_type = controller_type

    def get_model(self):
        return self.tip_velocity_estimator

    def get_tip_control(self, image):

        # baseline controller does not require image at all
        if self.controller_type == ControllerType.RELATIVE_POSITION_AND_ORIENTATION:
            relative_target_position = torch.unsqueeze(torch.tensor(
                self.target_object.get_position(relative_to=self.camera), dtype=torch.float32
            ), 0)
            relative_target_orientation = torch.unsqueeze(torch.tensor(
                self.target_object.get_orientation(relative_to=self.camera), dtype=torch.float32
            ), 0)
            control_input = torch.cat((relative_target_position, relative_target_orientation), dim=1)
            tip_control_single_batch = self.tip_velocity_estimator.predict(control_input)
            return tip_control_single_batch[0]

        h, w, _c = image.shape
        # select region of interest (manual crop or RL agent)
        image, pixels = self.roi_estimator.crop(image)
        # resizes image
        image = self.tip_velocity_estimator.resize_image(image)
        # apply normalisation and other transforms as required
        transformed_image = self.tip_velocity_estimator.transforms(image)

        with torch.no_grad():
            image_tensor = torch.unsqueeze(transformed_image, 0)
            # batch with single tip control command
            if self.controller_type == ControllerType.DEFAULT:
                tip_control_single_batch = self.tip_velocity_estimator.predict(image_tensor)

            elif self.controller_type == ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS:
                top_left_pixel = torch.unsqueeze(torch.tensor(pixels[1], dtype=torch.float32), 0)
                bottom_right_pixel = torch.unsqueeze(torch.tensor(pixels[4], dtype=torch.float32), 0)
                w_tensor = torch.tensor([w], dtype=torch.float32).unsqueeze(0)
                h_tensor = torch.tensor([h], dtype=torch.float32).unsqueeze(0)
                tip_control_single_batch = self.tip_velocity_estimator.predict((
                    image_tensor, top_left_pixel, bottom_right_pixel, w_tensor, h_tensor
                ))
            else:
                raise ValueError("Unrecognised controller type")

        tip_control = tip_control_single_batch[0]
        return tip_control
