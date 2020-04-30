import numpy as np
import torch

from lib.cv.utils import CvUtils


class ImageOffsetState(object):
    def __init__(self, data, x_center_previous=None, y_center_previous=None):
        """
        :param data: Dictionary with "image" with values in [-1.0, 1.0], stored in a PyTorch tensor,
        "tip_velocities" PyTorch tensor and "rotations" PyTorch tensor
        :param x_center_previous: X coordinate of center of crop from previous image
        :param y_center_previous: Y coordinate of center of crop from previous image
        """
        self.image = None
        self.data = data  # stored in State, but ignored for observation
        self.x_center_previous = x_center_previous
        self.y_center_previous = y_center_previous

        if (self.x_center_previous is None) + (self.y_center_previous is None) == 1:
            raise ValueError(
                "Either both x,y center coordinates of previous crop have to be provided, or none of them!"
            )
        if self.data is not None:
            image = self.data["image"]
            c, h, w = image.size()
            self.image = image.permute(1, 2, 0).numpy()  # convert to numpy
            if self.x_center_previous is None and self.y_center_previous is None:
                self.x_center_previous = int((w - 1) / 2)
                self.y_center_previous = int((h - 1) / 2)

    def get(self):
        # returns an observation (this state) as an np array
        center = np.array([self.x_center_previous, self.y_center_previous])
        flattened = self.image.flatten()
        return np.concatenate((center, flattened))

    def get_data(self):
        return self.data

    def get_np_image(self):
        # returns numpy image
        return self.image

    def get_tip_velocity(self):
        return self.data["tip_velocities"]

    def get_rotations(self):
        return self.data["rotations"]

    def get_center_crop(self):
        return self.x_center_previous, self.y_center_previous

    def set_center_crop(self, crop):
        self.x_center_previous, self.y_center_previous = crop

    def apply_action(self, data, dx, dy, cropped_width, cropped_height, restrict_crop_move=None):
        _c, height, width = self.data["image"].shape
        x_center_previous = self.x_center_previous + int(
            dx * ((min(width, restrict_crop_move) if restrict_crop_move is not None else width) - 1)
        )
        y_center_previous = self.y_center_previous + int(
            dy * ((min(height, restrict_crop_move) if restrict_crop_move is not None else height) - 1)
        )
        x_center_previous, y_center_previous = CvUtils.fit_crop_to_image(
            center_x=x_center_previous,
            center_y=y_center_previous,
            height=height,
            width=width,
            cropped_height=cropped_height,
            cropped_width=cropped_width
        )
        assert 0 <= x_center_previous < width and 0 <= y_center_previous < height
        return ImageOffsetState(
            data=data,
            x_center_previous=x_center_previous,
            y_center_previous=y_center_previous,
        )

    # for testing purposes more than anything, same for str and repr
    def __eq__(self, other):
        comparison = self.x_center_previous == other.x_center_previous and \
                     self.y_center_previous == other.y_center_previous
        are_data_none = (self.data is None) + (other.data is None)
        if are_data_none == 0:
            comparison = comparison and torch.allclose(self.data["image"], other.data["image"]) and \
                         np.allclose(self.data["tip_velocities"], other.data["tip_velocities"]) and \
                         np.allclose(self.data["rotations"], other.data["rotations"])

        return are_data_none != 1 and comparison

    def __str__(self):
        if self.data is None:
            return f"{self.x_center_previous}, {self.y_center_previous}"
        return f"{self.x_center_previous}, {self.y_center_previous}, image: {self.data['image']}, vel: {self.data['tip_velocities']}, rot: {self.data['rotations']}"

    def __repr__(self):
        return self.__str__()


class SpatialOffsetState(object):
    def __init__(self, spatial_features, image_offset_state):
        self.spatial_features = spatial_features  # numpy array in CPU
        # width and height are not determined by the spatial features, so we need to store them
        self.image_offset_state = image_offset_state

    def get(self):
        return self.spatial_features

    def get_tip_velocities(self):
        return self.image_offset_state.get_tip_velocities()

    def get_rotations(self):
        return self.image_offset_state.get_rotations()

    def get_np_image(self):
        return self.image_offset_state.get_np_image()

    def get_center_crop(self):
        return self.image_offset_state.get_center_crop()

    def apply_spatial_action(self, spatial_features, data, dx, dy, cropped_width, cropped_height, restrict_crop_move=None):
        image_offset_state = self.image_offset_state.apply_action(
            data=data, dx=dx, dy=dy, cropped_width=cropped_width, cropped_height=cropped_height,
            restrict_crop_move=restrict_crop_move
        )
        return SpatialOffsetState(
            spatial_features=spatial_features,
            image_offset_state=image_offset_state
        )


class FilterSpatialFeatureState(object):
    def __init__(self, k, spatial_features):
        # pick top k features
        self.k = k
        self.spatial_features = spatial_features.reshape(-1, 2)

    def get_top_k_features(self, action):
        # action is [-1.0, 1.0] ^ latent_dimension // 2
        feature_indices = np.argsort(action)[::-1][:self.k]
        return self.spatial_features[feature_indices].reshape(self.k * 2)
