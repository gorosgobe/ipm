import numpy as np


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
        return np.concatenate(np.array([self.x_center_previous, self.y_center_previous]), self.image.flatten())

    def get_np_image(self):
        # returns numpy image
        return self.image

    def get_tip_velocity(self):
        return self.data["tip_velocities"]

    def get_rotations(self):
        return self.data["rotations"]

    def apply_action(self, data, dx, dy):
        return State(
            data=data,
            x_center_previous=self.x_center_previous + dx,
            y_center_previous=self.y_center_previous + dy
        )

    @staticmethod
    def from_observation(observation):
        # returns a state from an observation
        pass
