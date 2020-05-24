import torch
from torchvision import transforms

from lib.common.utils import ResizeTransform
from lib.dsae.dsae import CoordinateUtils
from lib.soft.soft import SoftCNNLSTMNetwork, RecurrentFullImage, RecurrentCoordConv_32, RecurrentBaseline


class RNNTipVelocityControllerAdapter(object):
    def __init__(self, parameters, hidden_size, projection_scale, separate_prediction, keep_mask, version,
                 is_coord=False, size=(128, 96), gumbel_params=None):
        if version == "soft":
            self.model = SoftCNNLSTMNetwork(
                hidden_size=hidden_size,
                is_coord=is_coord,
                projection_scale=projection_scale,
                separate_prediction=separate_prediction,
                keep_masked=keep_mask,
                gumbel_params=gumbel_params
            )
        elif version == "full":
            self.model = RecurrentFullImage(
                hidden_size=hidden_size,
                is_coord=is_coord
            )
        elif version == "coordconv":
            self.model = RecurrentCoordConv_32(
                hidden_size=hidden_size
            )
            size = (32, 24)
        elif version == "baseline":
            self.model = RecurrentBaseline(
                hidden_size=hidden_size
            )
        else:
            raise ValueError("Unknown version for model!")

        self.version = version

        self.model.load_state_dict(parameters)
        self.is_coord = is_coord

        self.hidden_state = None
        self.demonstration_attention_maps = []

        self.resize_transform = ResizeTransform(size)
        self.input_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5]
            ) if version == "coordconv" else transforms.Normalize(
                [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            )
        ])

    def predict(self, image_batch):
        self.model.eval()
        with torch.no_grad():
            if self.version == "baseline":
                rels = image_batch.squeeze(0)
                action, self.hidden_state, _ = self.model(rels.unsqueeze(0).unsqueeze(0),
                                                          hidden_state=self.hidden_state)
                return action.squeeze(1)
            image = image_batch.squeeze(0)
            c, h, w = image.size()
            if c != 5 and self.is_coord:
                image_x, image_y = CoordinateUtils.get_image_coordinates(h, w, normalise=True)
                image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
                image_coordinates = image_coordinates.permute(2, 0, 1)
                # (2, H, W)
                image = torch.cat((image, image_coordinates), dim=0)
            # image is (c, h, w) but rnn takes (b, d_len, c, h, w)
            action, self.hidden_state, _ = self.model(image.unsqueeze(0).unsqueeze(0), hidden_state=self.hidden_state)
            # action is (b, d_len, 6)
            if self.version == "soft":
                self.demonstration_attention_maps.append(
                    (image_batch.squeeze(0), self.model.get_upsampled_attention().squeeze(0)))

        return action.squeeze(1)

    def start(self):
        # only required for recurrent controllers, use to initialise hidden_state
        self.hidden_state = None
        self.demonstration_attention_maps = []

    def get_np_attention_mapped_images(self):
        return self.get_np_attention_mapped_images_from(self.demonstration_attention_maps)

    @staticmethod
    def get_np_attention_mapped_images_from(demonstration_attention_maps, lower_weight=False):
        # if is_gumbel, we are using a gumbel softmax distribution to obtain a hard mask
        # need to soften the weights for this type of attention heatmap
        result = []
        for pair in demonstration_attention_maps:
            torch_img, torch_attention = pair
            np_img = (torch_img.permute(1, 2, 0).cpu().numpy() + 1) / 2
            attention_heatmap = torch_attention.permute(1, 2, 0).cpu().numpy()
            res = np_img * 0.7 + attention_heatmap * (3 if not lower_weight else 0.3)
            result.append(res)
        return result

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
            hidden_size=info["hidden_size"],
            projection_scale=info["projection_scale"],
            is_coord=info["is_coord"] if "is_coord" in info else False,
            separate_prediction=info["separate_prediction"] if "separate_prediction" in info else True,
            keep_mask=info["keep_mask"] if "keep_mask" in info else False,
            version=info["version"] if "version" in info else "soft",
            gumbel_params=info["gumbel_params"] if "gumbel_params" in info else None
        )
