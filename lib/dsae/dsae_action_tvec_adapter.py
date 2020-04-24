import torch

from lib.common.utils import ResizeTransform


class DSAETipVelocityEstimatorAdapter(object):
    def __init__(self, feature_provider, action_predictor, size=(128, 96)):
        self.feature_provider = feature_provider
        self.action_predictor = action_predictor
        self.resize_transform = ResizeTransform(size)

    def predict(self, image_batch):
        self.feature_provider.eval()
        self.action_predictor.eval()
        with torch.no_grad():
            features = self.feature_provider(image_batch)
            action = self.action_predictor(features)

        return action

    def resize_image(self, image):
        # resize to input image, (96, 128)
        return self.resize_transform(image)
