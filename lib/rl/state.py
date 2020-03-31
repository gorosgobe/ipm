class State(object):
    def __init__(self, image, x_center_previous=None, y_center_previous=None):
        """
        :param image: Image with values in [-1, 1], stored in a PyTorch tensor.
        :param x_center_previous: X coordinate of center of crop from previous image
        :param y_center_previous: Y coordinate of center of crop from previous image
        """
        self.image = image
        c, h, w = self.image.size()
        self.x_center_previous = x_center_previous
        self.y_center_previous = y_center_previous

        if (self.x_center_previous is None) + (self.y_center_previous is None) == 1:
            raise ValueError(
                "Either both x,y center coordinates of previous crop have to be provided, or none of them!")

        if self.x_center_previous is None and self.y_center_previous is None:
            self.x_center_previous = int(w / 2)
            self.y_center_previous = int(h / 2)

    def get_observation(self):
        # returns an observation (this state) as an np array
        pass

    def get_image(self):
        # returns torch image
        return self.image

    @staticmethod
    def from_observation(observation):
        # returns a state from an observation
        pass
