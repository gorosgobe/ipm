import enum

import torch
from torch import nn
from torch.nn import Sequential
from torchmeta.modules import MetaModule

from lib.stn.linearized_multisampling_release.warp.linearized import LinearizedMutilSampler


class STN_SamplingType(enum.Enum):
    DEFAULT_BILINEAR = 0
    LINEARISED = 1


class SpatialLocalisation(nn.Module):
    # TODO: implement, uses normal CNN but transforms them into spatial features
    pass


# TODO: remove
class FPN(nn.Module):
    def __init__(self, add_coord=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=5 if add_coord else 3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2)
        self.batch_norm3 = nn.BatchNorm2d(256)
        self.activ = nn.ReLU()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv3_1x1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1)
        self.conv2_1x1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv1_1x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)

        self.batch_norm1_reduced = nn.BatchNorm2d(64)
        self.batch_norm2_reduced = nn.BatchNorm2d(64)
        self.batch_norm3_reduced = nn.BatchNorm2d(64)

        self.batch_norm_final1 = nn.BatchNorm2d(16)
        self.final1 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=2)
        self.batch_norm_final2 = nn.BatchNorm2d(16)
        self.final2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        out_conv1 = self.activ(self.batch_norm1(self.conv1(x)))
        out_conv2 = self.activ(self.batch_norm2(self.conv2(out_conv1)))
        out_conv3 = self.activ(self.batch_norm3(self.conv3(out_conv2)))

        out_conv3_reduced = self.activ(self.batch_norm3_reduced(self.conv3_1x1(out_conv3)))
        out_conv3_up = self.upsample(out_conv3_reduced)
        out_conv2_reduced = self.activ(self.batch_norm2_reduced(self.conv2_1x1(out_conv2)))
        out_3_to_2 = out_conv2_reduced + out_conv3_up

        out_3_to_2_up = self.upsample(out_3_to_2)
        out_conv1_reduced = self.activ(self.batch_norm1_reduced(self.conv1_1x1(out_conv1)))
        out_2_to_1 = out_conv1_reduced + out_3_to_2_up

        out = self.activ(self.batch_norm_final1(self.final1(out_2_to_1)))
        out = self.activ(self.batch_norm_final2(self.final2(out)))

        out_flattened = self.flatten(out)
        return out_flattened


class LocalisationParamRegressor(nn.Module):
    def __init__(self, add_coord=True, scale=None):
        # if scale is None, learn it. Otherwise it must be a float value for the scale
        super().__init__()
        self.scale = scale
        self.cnn_model = Sequential(
            nn.Conv2d(in_channels=5 if add_coord else 3, out_channels=64, kernel_size=7, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.fc_model = Sequential(
            nn.Linear(in_features=480, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=3) if scale is None else nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, x):
        """
        Returns attention affine transform
        [ s 0 t_x]
        [ 0 s t_y]
        """
        # TODO: abstract this into a separate part, so we can plug a different one
        out_cnn_model = self.cnn_model(x)
        res = self.fc_model(out_cnn_model)
        # res (B, 3)
        scale_position = torch.tensor([[1, 0, 0, 0, 1, 0]]).to(x.device)
        t_x_position = torch.tensor([[0, 0, 1, 0, 0, 0]]).to(x.device)
        t_y_position = torch.tensor([[0, 0, 0, 0, 0, 1]]).to(x.device)
        if self.scale is None:
            s = res[:, 0].unsqueeze(-1)
            t_x = res[:, 1].unsqueeze(-1)
            t_y = res[:, 2].unsqueeze(-1)
            out = s * scale_position + t_x * t_x_position + t_y * t_y_position
        else:
            t_x = res[:, 0].unsqueeze(-1)
            t_y = res[:, 1].unsqueeze(-1)
            out = self.scale * scale_position + t_x * t_x_position + t_y * t_y_position
        # out (B, 6)
        # learn offset
        return out + torch.tensor([[1, 0, 0, 0, 1, 0]], dtype=torch.float32).to(x.device)


class SpatialTransformerNetwork(MetaModule):
    def __init__(self, localisation_param_regressor, model, output_size,
                 sampling_type=STN_SamplingType.DEFAULT_BILINEAR):
        super().__init__()
        self.localisation_param_regressor = localisation_param_regressor
        self.model = model
        # output_size (w_p, h_p)
        self.w, self.h = output_size
        self.sampling_type = sampling_type
        self.transformed_image = None
        self.transformation_params = None

    def forward(self, x, params=None):
        if isinstance(x, tuple):
            image_batch, _, _, _, _ = x
        else:
            image_batch = x
        b, c, h, w = image_batch.size()
        transformation_params = self.localisation_param_regressor(image_batch)
        # transformation_params (b, 2 * 3)
        transformation_params = transformation_params.view(b, 2, 3)
        grid = nn.functional.affine_grid(transformation_params, (b, c, h, w), align_corners=False)
        if self.sampling_type == STN_SamplingType.DEFAULT_BILINEAR:
            image_batch = nn.functional.grid_sample(image_batch, grid, align_corners=False)
        else:
            image_batch = LinearizedMutilSampler.linearized_grid_sample(image_batch, grid, "zeros")

        self.transformation_params = transformation_params
        # so we can access it, to plot
        self.transformed_image = image_batch
        # downsample to coordconv size
        image_batch = nn.functional.interpolate(image_batch, mode="bilinear", size=(self.h, self.w), align_corners=True)
        return self.model(image_batch, params=params)
