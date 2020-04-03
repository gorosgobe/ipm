import numpy as np
import torch


class State(object):
    def __init__(self, data, x_center_previous=None, y_center_previous=None):
        """
        :param data: Dictionary with "image" with values in [-1.0, 1.0], stored in a PyTorch tensor,
        "tip_velocities" PyTorch tensor and "rotations" PyTorch tensor
        :param x_center_previous: X coordinate of center of crop from previous image
        :param y_center_previous: Y coordinate of center of crop from previous image
        """
        image = data["image"]
        c, h, w = image.size()
        print("State created", x_center_previous, y_center_previous)
        self.image = image.permute(1, 2, 0).numpy()  # convert to numpy
        self.data = data  # stored in State, but ignored for observation
        self.x_center_previous = x_center_previous
        self.y_center_previous = y_center_previous

        if (self.x_center_previous is None) + (self.y_center_previous is None) == 1:
            raise ValueError(
                "Either both x,y center coordinates of previous crop have to be provided, or none of them!")

        if self.x_center_previous is None and self.y_center_previous is None:
            self.x_center_previous = int(w / 2)
            self.y_center_previous = int(h / 2)

    def get(self):
        # returns an observation (this state) as an np array
        center = np.array([self.x_center_previous, self.y_center_previous])
        flattened = self.image.flatten()
        return np.concatenate((center, flattened))

    def get_np_image(self):
        # returns numpy image
        return self.image

    def get_tip_velocity(self):
        return self.data["tip_velocities"]

    def get_rotations(self):
        return self.data["rotations"]

    def get_center_crop(self):
        return self.x_center_previous, self.y_center_previous

    def apply_action(self, data, dx, dy):
        print("DX", dx)
        print("DY", dy)
        print("x center previous", self.x_center_previous)
        print("y center_previous", self.y_center_previous)
        return State(
            data=data,
            x_center_previous=self.x_center_previous + dx,
            y_center_previous=self.y_center_previous + dy
        )

    @staticmethod
    def from_observation(observation):
        # returns a state from an observation
        pass

    def __eq__(self, other):
        return torch.allclose(self.data["image"], other.data["image"]) and \
               np.allclose(self.data["tip_velocities"], other.data["tip_velocities"]) and \
               np.allclose(self.data["rotations"], other.data["rotations"]) and \
               self.x_center_previous == other.x_center_previous and \
               self.y_center_previous == other.y_center_previous

    def __str__(self):
        return f"{self.x_center_previous}, {self.y_center_previous}, image: {self.data['image']}, vel: {self.data['tip_velocities']}, rot: {self.data['rotations']}"

    def __repr__(self):
        return str(self)