import torch
import torch.nn.functional as F

from lib.cv.controller import SpatialDimensionAdder
from lib.cv.pos_enc import PositionalEncodings
from senet_pytorch.senet.se_module import SELayer


class AttentionNetworkTile(torch.nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
        # spatial information is encoded as a tiled feature map, added to conv2
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

        out_conv2 = F.relu(self.batch_norm2.forward(out_conv2_tiled))
        out_conv3 = F.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = F.relu(self.fc1.forward(out_conv3))
        out_fc2 = F.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3

    def normalise(self, pixel_batch, original_image_width, original_image_height):
        w, h = pixel_batch.split((1, 1), dim=1)
        w = w / original_image_width
        h = h / original_image_height
        return torch.cat((w, h), dim=1)

    def get_tiled_spatial_info(self, h, w, tl, br):
        self.assert_size(h, w)
        top_left_tile = tl.repeat(1, w // 2).unsqueeze(1).repeat(1, h, 1).unsqueeze(1)
        bottom_right_tile = br.repeat(1, w // 2).unsqueeze(1).repeat(1, h, 1).unsqueeze(1)
        return torch.cat((top_left_tile, bottom_right_tile), dim=1)

    def assert_size(self, h, w):
        assert h == 10 and w == 14  # for 64x48 input


class AttentionNetworkTile_32(AttentionNetworkTile):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        self.fc1 = torch.nn.Linear(in_features=32, out_features=64)

    def assert_size(self, h, w):
        assert h == 4 and w == 6  # for 32x24 input


class AttentionNetworkCoord(torch.nn.Module):
    def __init__(self, _image_width, _image_height, add_se_blocks=False, reduction=16):
        super().__init__()
        self.add_se_blocks = add_se_blocks
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
        if self.add_se_blocks:
            self.se_1 = SELayer(channel=64, reduction=reduction)
            self.se_2 = SELayer(channel=32, reduction=reduction)
            self.se_3 = SELayer(channel=16, reduction=reduction)

    def forward(self, x):
        if isinstance(x, tuple):
            image_batch, top_left_pixel, bottom_right_pixel, original_image_width, original_image_height = x
        else:
            image_batch = x
        batch_size = image_batch.size()[0]
        out_conv1 = F.relu(self.batch_norm1.forward(self.conv1.forward(image_batch)))
        if self.add_se_blocks:
            out_conv1 = self.se_1(out_conv1)
        out_conv2 = F.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        if self.add_se_blocks:
            out_conv2 = self.se_2(out_conv2)
        out_conv3 = F.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        if self.add_se_blocks:
            out_conv3 = self.se_3(out_conv3)
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = F.relu(self.fc1.forward(out_conv3))
        out_fc2 = F.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3


class AttentionNetworkCoord_32(AttentionNetworkCoord):
    def __init__(self, image_width, image_height, add_se_blocks=False, reduction=16):
        super().__init__(image_width, image_height, add_se_blocks=add_se_blocks, reduction=reduction)
        self.fc1 = torch.nn.Linear(in_features=32, out_features=64)


class AttentionNetworkCoordSE(AttentionNetworkCoord):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height, add_se_blocks=True)


class AttentionNetworkCoordSE_32(AttentionNetworkCoord_32):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height, add_se_blocks=True)


class AttentionNetworkCoordRot(AttentionNetworkCoord):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        # coord 64, but with one more input feature map
        self.conv1 = torch.nn.Conv2d(in_channels=6, out_channels=64, kernel_size=5, stride=2, padding=1)


class AttentionNetworkCoordRot_32(AttentionNetworkCoord_32):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        self.conv1 = torch.nn.Conv2d(in_channels=6, out_channels=64, kernel_size=5, stride=2, padding=1)


# Crop is encoded via positional encodings
class AttentionNetworkPos(torch.nn.Module):
    def __init__(self, _image_width, _image_height, pos_dimension):
        # pos_dimension is dimension of encoding of each pixel coordinate (pos_dimension dimensions for x, pos_dimension dimensions for y)
        super().__init__()
        self.pos_dimension = pos_dimension
        # spatial information as positional encodings
        self.conv1 = torch.nn.Conv2d(in_channels=3 + 2 * 2 * pos_dimension, out_channels=64, kernel_size=5, stride=2,
                                     padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(in_features=384, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    @staticmethod
    def create(pos_dimension):
        return lambda i_w, im_h, p=pos_dimension: AttentionNetworkPos(i_w, im_h, pos_dimension=p)

    def forward(self, x):
        if isinstance(x, tuple):
            image_batch, top_left_pixel, bottom_right_pixel, original_image_width, original_image_height = x
        else:
            image_batch = x
        batch_size, c, h, w = image_batch.size()
        # image has coordconv channels, need to transform the position in -1, 1 to the positional encoding
        assert c == 5
        image_rgb_batch, image_coord_batch = torch.split(image_batch, (3, 2), dim=1)
        pos_maps = PositionalEncodings.get_positional_encodings(L=self.pos_dimension,
                                                                batched_coord_maps=image_coord_batch)
        image_batch_with_pos = torch.cat((image_rgb_batch, pos_maps), dim=1)
        out_conv1 = F.relu(self.batch_norm1.forward(self.conv1.forward(image_batch_with_pos)))
        out_conv2 = F.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        out_conv3 = F.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = F.relu(self.fc1.forward(out_conv3))
        out_fc2 = F.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3


class AttentionNetworkPos_32(AttentionNetworkPos):
    def __init__(self, image_width, image_height, pos_dimension=1):
        super().__init__(image_width, image_height)
        self.fc1 = torch.nn.Linear(in_features=32, out_features=64)

    @staticmethod
    def create(pos_dimension):
        return lambda i_w, im_h, p=pos_dimension: AttentionNetworkPos_32(i_w, im_h, pos_dimension=p)


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
        w = w / (original_image_width - 1)
        h = h / (original_image_height - 1)
        return torch.cat((w, h), dim=1)


class AttentionNetworkV2_32(AttentionNetworkV2):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        self.fc1 = torch.nn.Linear(in_features=32, out_features=60)


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


# for 32 x 24 images, override flattening layer
# to make sure that input is correct, rather than writing generic AttentionNetwork
# better be safe than sorry (happened too many times now), at the expense of some code duplication
class AttentionNetwork_32(AttentionNetwork):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        # redefine flattening layer
        self.fc1 = torch.nn.Linear(in_features=32, out_features=62)


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


class BaselineSimilarParamsAttentionCoord64(torch.nn.Module):
    def __init__(self, _image_width, _image_height):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=6, out_features=271)
        self.fc2 = torch.nn.Linear(in_features=271, out_features=271)
        self.fc3 = torch.nn.Linear(in_features=271, out_features=271)
        self.fc4 = torch.nn.Linear(in_features=271, out_features=6)

    def forward(self, x):
        out_fc1 = F.relu(self.fc1.forward(x))
        out_fc2 = F.relu(self.fc2.forward(out_fc1))
        out_fc3 = F.relu(self.fc3.forward(out_fc2))
        out_fc4 = self.fc4.forward(out_fc3)
        return out_fc4


class FullImageNetwork(torch.nn.Module):
    """
    Removes max pool layers, applying only stride, network for full image, predicting
    velocities and orientation change.
    """

    def __init__(self, image_width, image_height):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
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


class FullImageNetworkCoord(FullImageNetwork):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        self.conv1 = torch.nn.Conv2d(in_channels=5, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.spatial_dimension_adder = SpatialDimensionAdder()

    def forward(self, x):
        b, c, h, w = x.size()
        assert self.image_width == w and self.image_height == h
        i_tensor, j_tensor = self.spatial_dimension_adder.get_tensor_batch_spatial_dimensions(b, h, w)
        image_batch_spatial = torch.cat((x, i_tensor, j_tensor), dim=1)
        return super().forward(image_batch_spatial)


class FullImageNetwork_64(FullImageNetwork):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        self.fc1 = torch.nn.Linear(in_features=384, out_features=64)


class FullImageNetworkCoord_64(FullImageNetwork_64):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        self.conv1 = torch.nn.Conv2d(in_channels=5, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.spatial_dimension_adder = SpatialDimensionAdder()

    def forward(self, x):
        b, c, h, w = x.size()
        assert self.image_width == w and self.image_height == h
        i_tensor, j_tensor = self.spatial_dimension_adder.get_tensor_batch_spatial_dimensions(b, h, w)
        image_batch_spatial = torch.cat((x, i_tensor, j_tensor), dim=1)
        return super().forward(image_batch_spatial)


class FullImageNetwork_32(FullImageNetwork):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        self.fc1 = torch.nn.Linear(in_features=32, out_features=64)


class FullImageNetworkCoord_32(FullImageNetwork_32):
    def __init__(self, image_width, image_height):
        super().__init__(image_width, image_height)
        self.conv1 = torch.nn.Conv2d(in_channels=5, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.spatial_dimension_adder = SpatialDimensionAdder()

    def forward(self, x):
        b, c, h, w = x.size()
        assert self.image_width == w and self.image_height == h
        i_tensor, j_tensor = self.spatial_dimension_adder.get_tensor_batch_spatial_dimensions(b, h, w)
        image_batch_spatial = torch.cat((x, i_tensor, j_tensor), dim=1)
        return super().forward(image_batch_spatial)


class FullSoftImageNetwork(torch.nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.soft1 = SoftBlock(in_channels=64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.soft2 = SoftBlock(in_channels=32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        self.soft3 = SoftBlock(in_channels=16)
        self.fc1 = torch.nn.Linear(in_features=2240, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        batch_size = x.size()[0]
        out_conv1 = self.soft1(F.relu(self.batch_norm1.forward(self.conv1.forward(x))))
        out_conv2 = self.soft2(F.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1))))
        out_conv3 = self.soft3(F.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2))))
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = F.relu(self.fc1.forward(out_conv3))
        out_fc2 = F.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3


class SoftBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.key_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.key_batch_norm = torch.nn.BatchNorm2d(in_channels)
        self.query_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.query_batch_norm = torch.nn.BatchNorm2d(in_channels)
        self.values_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.values_batch_norm = torch.nn.BatchNorm2d(in_channels)
        self.softmax = torch.nn.Softmax2d()

    def forward(self, x):
        # x is (B, C, H, W)
        key_proj = self.key_batch_norm(self.key_layer(x))
        query_proj = self.query_batch_norm(self.query_layer(x))
        values_proj = self.values_batch_norm(self.values_layer(x))
        attention_map = self.softmax(key_proj * query_proj)
        attended_output_diff = attention_map * values_proj
        assert x.size() == attended_output_diff.size()
        return x + attended_output_diff


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


if __name__ == '__main__':
    a = FullImageNetworkCoord_64(64, 48)
    t = torch.rand((2, 3, 48, 64))
    res = a(t)
