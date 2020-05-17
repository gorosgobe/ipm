import enum

import torch
from torch import nn

from lib.stn.linearized_multisampling_release.warp.linearized import LinearizedMutilSampler


class STN_SamplingType(enum.Enum):
    DEFAULT_BILINEAR = 0
    LINEARISED = 1


class LocalisationParamRegressor(nn.Module):
    def __init__(self, add_coord=True, scale=None):
        # if scale is None, learn it. Otherwise it must be a float value for the scale
        super().__init__()
        self.scale = scale
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=5 if add_coord else 3, out_channels=64, kernel_size=7, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=480, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=3) if scale is None else nn.Linear(in_features=64, out_features=2)
        )

        if self.scale is None:
            self.model[12].weight.data.zero_()
            self.model[12].bias.data.copy_(torch.tensor([1, 0, 0], dtype=torch.float))
        else:
            self.model[12].weight.data.zero_()
            self.model[12].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))

    def forward(self, x):
        """
        :param x: Image batch
        :return: Attention affine transform
        [ s 0 t_x]
        [ 0 s t_y]
        """
        res = self.model(x)
        # res (B, 3)
        if self.scale is None:
            s = res[:, 0].unsqueeze(-1)
            t_x = res[:, 1].unsqueeze(-1)
            t_y = res[:, 2].unsqueeze(-1)
            position = torch.tensor([[1, 0, 2, 0, 1, 3]]).to(x.device)
            scale_position = (position == 1).float()
            t_x_position = (position == 2).float()
            t_y_position = (position == 3).float()
            out = s * scale_position + t_x * t_x_position + t_y * t_y_position
        else:
            t_x = res[:, 0].unsqueeze(-1)
            t_y = res[:, 1].unsqueeze(-1)
            position = torch.tensor([[self.scale, 0, 1, 0, self.scale, 2]]).to(x.device)
            t_x_position = (position == 1).float()
            t_y_position = (position == 2).float()
            out = t_x * t_x_position + t_y * t_y_position
        # out (B, 6)
        return out


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, localisation_param_regressor, model, output_size,
                 sampling_type=STN_SamplingType.DEFAULT_BILINEAR):
        super().__init__()
        self.localisation_param_regressor = localisation_param_regressor
        self.model = model
        # output_size (w_p, h_p)
        self.w, self.h = output_size
        self.sampling_type = sampling_type
        self.transformed_image = None

    def forward(self, x):
        if isinstance(x, tuple):
            image_batch, _, _, _, _ = x
        else:
            image_batch = x
        b, c, h, w = image_batch.size()
        transformation_params = self.localisation_param_regressor(image_batch)
        # transformation_params (b, 2 * 3)
        transformation_params = transformation_params.view(b, 2, 3)
        grid = nn.functional.affine_grid(transformation_params, (b, c, self.h, self.w), align_corners=True)
        if self.sampling_type == STN_SamplingType.DEFAULT_BILINEAR:
            image_batch = nn.functional.grid_sample(image_batch, grid, align_corners=True)
        else:
            image_batch = LinearizedMutilSampler.linearized_grid_sample(image_batch, grid, "zeros")


        # so we can access it, to plot
        self.transformed_image = image_batch
        return self.model(image_batch)
