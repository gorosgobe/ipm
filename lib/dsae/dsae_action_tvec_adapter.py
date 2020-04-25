import torch
from torchvision import transforms

from lib.common.utils import ResizeTransform


class DSAETipVelocityEstimatorAdapter(object):
    def __init__(self, feature_provider, action_predictor, size=(128, 96)):
        self.feature_provider = feature_provider
        self.action_predictor = action_predictor
        self.resize_transform = ResizeTransform(size)
        self.input_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def predict(self, image_batch):
        self.feature_provider.eval()
        self.action_predictor.eval()
        with torch.no_grad():
            features = self.feature_provider(image_batch)
            b = features.size()[0]
            action = self.action_predictor(features.view(b, -1))

        return action

    def resize_image(self, image):
        # resize to input image, (96, 128)
        return self.resize_transform(image)

    def transforms(self, image):
        return self.input_transforms(image)
