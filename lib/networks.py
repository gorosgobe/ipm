import torch
import torch.nn.functional as F


class AttentionNetworkTile(torch.nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
        # spatial information is encoded as a tiled feature map, added to conv2d
        # from https://arxiv.org/pdf/1610.00696.pdf
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(34)
        self.conv3 = torch.nn.Conv2d(in_channels=34, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(in_features=384, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        image_batch, top_left_pixel, bottom_right_pixel, original_image_width, original_image_height = x
        batch_size = image_batch.size()[0]
        out_conv1 = F.relu(self.batch_norm1.forward(self.conv1.forward(image_batch)))
        # tile pixel information
        out_conv2_untiled = self.conv2.forward(out_conv1)
        _b, _c, h, w = out_conv2_untiled.size()
        tl = self.normalise(top_left_pixel, original_image_width, original_image_height)
        br = self.normalise(bottom_right_pixel, original_image_width, original_image_height)
        out_conv2_tiled = torch.cat((out_conv2_untiled, self.get_tiled_spatial_info(h, w, tl, br)), dim=1)

        out_conv2 = torch.nn.functional.relu(self.batch_norm2.forward(out_conv2_tiled))
        out_conv3 = torch.nn.functional.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = torch.nn.functional.relu(self.fc1.forward(out_conv3))
        out_fc2 = torch.nn.functional.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3

    def normalise(self, pixel_batch, original_image_width, original_image_height):
        w, h = pixel_batch.split((1, 1), dim=1)
        w = w / original_image_width
        h = h / original_image_height
        return torch.cat((w, h), dim=1)

    def get_tiled_spatial_info(self, h, w, tl, br):
        assert h == 10 and w == 14  # for 64x48 input
        top_left_tile = tl.repeat(1, w // 2).unsqueeze(1).repeat(1, h, 1).unsqueeze(1)
        bottom_right_tile = br.repeat(1, w // 2).unsqueeze(1).repeat(1, h, 1).unsqueeze(1)
        return torch.cat((top_left_tile, bottom_right_tile), dim=1)


class AttentionNetworkCoord(torch.nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
        # spatial information is encoded as coord feature maps, one for x and one for y dimensions, fourth/fifth channels
        self.conv1 = torch.nn.Conv2d(in_channels=5, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(in_features=384, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        image_batch, top_left_pixel, bottom_right_pixel, original_image_width, original_image_height = x
        batch_size = image_batch.size()[0]
        out_conv1 = F.relu(self.batch_norm1.forward(self.conv1.forward(image_batch)))
        out_conv2 = torch.nn.functional.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        out_conv3 = torch.nn.functional.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = torch.nn.functional.relu(self.fc1.forward(out_conv3))
        out_fc2 = torch.nn.functional.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3


class AttentionNetworkV2(torch.nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(in_features=384, out_features=60)
        # add x, y position of top left and bottom right pixels, normalised, of bounding box
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        image_batch, top_left_pixel, bottom_right_pixel, original_image_width, original_image_height = x
        batch_size = image_batch.size()[0]
        out_conv1 = F.relu(self.batch_norm1.forward(self.conv1.forward(image_batch)))
        out_conv2 = F.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        out_conv3 = F.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = F.relu(self.fc1.forward(out_conv3))

        top_left_pixel = self.normalise(top_left_pixel, original_image_width, original_image_height)
        bottom_right_pixel = self.normalise(bottom_right_pixel, original_image_width, original_image_height)

        cropped_concat = torch.cat((out_fc1, top_left_pixel, bottom_right_pixel), dim=1)
        out_fc2 = F.relu(self.fc2.forward(cropped_concat))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3

    def normalise(self, pixel_batch, original_image_width, original_image_height):
        w, h = pixel_batch.split((1, 1), dim=1)
        w = w / original_image_width
        h = h / original_image_height
        return torch.cat((w, h), dim=1)


class AttentionNetworkV3(AttentionNetworkV2):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        self.image_width = image_width
        self.image_height = image_height

    def normalise(self, pixel_batch, original_image_width, original_image_height):
        """
        This version normalises so pixels provided are in the range of the cropped version of the image
        In contrast, V2 provides two values between 0 and 1. In this case, we provide two values between 0 and
        self.image_width/self.image_height, respectively (cropped image)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_width_tensor = torch.tensor(
            [self.image_width],
            dtype=torch.float32,
            device=device
        ).unsqueeze(0).expand(original_image_width.size())
        image_height_tensor = torch.tensor(
            [self.image_height],
            dtype=torch.float32,
            device=device
        ).unsqueeze(0).expand(original_image_height.size())
        w, h = pixel_batch.split((1, 1), dim=1)
        w = w / (original_image_width / image_width_tensor)
        h = h / (original_image_height / image_height_tensor)
        return torch.cat((w, h), dim=1)


class AttentionNetwork(torch.nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(in_features=384, out_features=62)
        # add x, y position of top left pixel of bounding box
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        image_batch, top_left_pixel, _, _, _ = x
        batch_size = image_batch.size()[0]
        out_conv1 = F.relu(self.batch_norm1.forward(self.conv1.forward(image_batch)))
        out_conv2 = F.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        out_conv3 = F.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = F.relu(self.fc1.forward(out_conv3))
        cropped_concat = torch.cat((out_fc1, top_left_pixel), dim=1)
        out_fc2 = F.relu(self.fc2.forward(cropped_concat))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3


class BaselineNetwork(torch.nn.Module):
    def __init__(self, _image_width, _image_height):
        """
        Similar to FullImageNetwork, but takes only relative positions as target position estimate
        :param image_width: Unused
        :param image_height: Unused
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=6, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        out_fc1 = F.relu(self.fc1.forward(x))
        out_fc2 = F.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3


class FullImageNetwork(torch.nn.Module):
    """
    Removes max pool layers, applying only stride, network for full image, predicting
    velocities and orientation change.
    """

    def __init__(self, image_width, image_height):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(in_features=2240, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        batch_size = x.size()[0]
        out_conv1 = F.relu(self.batch_norm1.forward(self.conv1.forward(x)))
        out_conv2 = F.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        out_conv3 = F.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = F.relu(self.fc1.forward(out_conv3))
        out_fc2 = F.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3


class Network(torch.nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(in_features=240, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=3)

    def forward(self, x):
        batch_size = x.size()[0]
        out_conv1 = F.relu(self.batch_norm1.forward(self.conv1.forward(x)))
        out_conv1 = self.pool1.forward(out_conv1)
        out_conv2 = F.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        out_conv2 = self.pool2.forward(out_conv2)
        out_conv3 = F.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = self.pool3.forward(out_conv3)
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = F.relu(self.fc1.forward(out_conv3))
        out_fc2 = self.fc2.forward(out_fc1)
        return out_fc2
