import torch
from tip_velocity_estimator import TipVelocityEstimator

class TipVelocityController(object):
    def __init__(self, model_location):
        self.tip_velocity_estimator = TipVelocityEstimator.load(model_location)

    def get_tip_velocity(self, image):
        image = self.tip_velocity_estimator.resize_image(image)
        # apply normalisation and other transforms as required
        transformed_image = self.tip_velocity_estimator.transforms(image)
        with torch.no_grad():
            image_tensor = torch.unsqueeze(transformed_image, 0)
            # batch with single tip velocity
            tip_velocity_single_batch = self.tip_velocity_estimator.predict(image_tensor)
            tip_velocity = tip_velocity_single_batch[0]

        return tip_velocity

