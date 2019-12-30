import torch
from tip_velocity_estimator import TipVelocityEstimator

class IdentityCropper(object):
    def crop(self, image):
        return image

class CenterCropper(object):
    def __init__(self, half_size_height, half_size_width):
        self.half_size_height = half_size_height
        self.half_size_width = half_size_width

    def crop(self, image):
        height, width, _ = image.shape
        center_x = int(width / 2)
        center_y = int(height / 2)
        cropped_image = image[
                        center_y - self.half_size_height:center_y + self.half_size_height,
                        center_x - self.half_size_width:center_x + self.half_size_width
                        ]
        return cropped_image


class TipVelocityController(object):
    def __init__(self, tve_model_location, roi_estimator):
        self.tip_velocity_estimator = TipVelocityEstimator.load(tve_model_location)
        self.roi_estimator = roi_estimator

    def get_model(self):
        return self.tip_velocity_estimator

    def get_tip_velocity(self, image):
        # select region of interest (manual crop or RL agent)
        # TODO: this might need to be reordered, TBD
        image = self.roi_estimator.crop(image)
        # resizes image
        image = self.tip_velocity_estimator.resize_image(image)
        # apply normalisation and other transforms as required
        transformed_image = self.tip_velocity_estimator.transforms(image)
        with torch.no_grad():
            image_tensor = torch.unsqueeze(transformed_image, 0)
            # batch with single tip velocity
            tip_velocity_single_batch = self.tip_velocity_estimator.predict(image_tensor)
            tip_velocity = tip_velocity_single_batch[0]

        return tip_velocity
