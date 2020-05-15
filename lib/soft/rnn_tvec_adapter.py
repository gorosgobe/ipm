import torch
from torchvision import transforms

from lib.soft.soft import SoftCNNLSTMNetwork
from lib.common.utils import ResizeTransform


class RNNTipVelocityControllerAdapter(object):
    def __init__(self, parameters, hidden_size, size=(128, 96)):
        self.model = SoftCNNLSTMNetwork(hidden_size=hidden_size)
        self.model.load_state_dict(parameters)
        self.hidden_state = None
        self.resize_transform = ResizeTransform(size)
        self.input_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def predict(self, image_batch):
        self.model.eval()
        with torch.no_grad():
            # image batch is (b, c, h, w) but rnn takes (b, d_len, c, h, w)
            action, self.hidden_state = self.model(image_batch.unsqueeze(1), hidden_state=self.hidden_state)
            # action is (b, d_len, 6)

        return action.squeeze(1)

    def start(self):
        # only required for recurrent controllers, use to initialise hidden_state
        self.hidden_state = None

    def resize_image(self, image):
        # resize to input image, (96, 128)
        return self.resize_transform(image)

    def transforms(self, image):
        return self.input_transforms(image)

    @staticmethod
    def load(path):
        info = torch.load(path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        return RNNTipVelocityControllerAdapter(
            parameters=info["state_dict"],
            hidden_size=info["hidden_size"]
        )
