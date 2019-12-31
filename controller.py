import torch
from tip_velocity_estimator import TipVelocityEstimator


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
