import torch
from torchvision import transforms

from lib.common.utils import ResizeTransform
from lib.stn.stn_visualise import visualise


class STNControllerAdapter(object):
    def __init__(self, stn):
        self.stn = stn
        self.resize_transform = ResizeTransform((128, 96))
        self.input_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5])
        ])
        self.images = []

    def predict(self, image_batch):
        self.stn.eval()
        with torch.no_grad():
            action = self.stn(image_batch)
            drawn_image = visualise(
                name="",  # does not matter, saved outside
                model=self.stn,
                image_batch=image_batch,
                return_drawn_image=True
            )
            drawn_image = (drawn_image + 1) / 2
            self.images.append(drawn_image.squeeze(0).numpy().transpose(1, 2, 0))
        return action

    def get_images(self):
        return self.images

    def start(self):
        self.images = []

    def resize_image(self, image):
        # resize to input image, (96, 128)
        return self.resize_transform(image)

    def transforms(self, image):
        # ROI will be a cropper that does not change the size of the numpy image
        # so here we just need to normalise
        return self.input_transforms(image)
