import enum

import torch
from torch import nn
from torch.nn import Sequential
from torchmeta.modules import MetaModule

from lib.stn.linearized_multisampling_release.warp.linearized import LinearizedMutilSampler


class STN_SamplingType(enum.Enum):
    DEFAULT_BILINEAR = 0
    LINEARISED = 1


class SpatialLocalisationRegressor(nn.Module):
    def __init__(self, dsae, latent_dimension, scale=None):
        super().__init__()
        self.dsae = dsae
        self.scale = scale
        self.fc_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=latent_dimension, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3 if scale is None else 2)
        )

        self.fc_model[-1].weight.data.zero_()
        if scale is None:
            self.fc_model[-1].bias.data.copy_(torch.tensor([1, 0, 0], dtype=torch.float))
        else:
            self.fc_model[-1].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))

    def forward(self, x):
        spatial_features = self.dsae(x[:, :3])
        res = self.fc_model(spatial_features)
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
        return out


class LocalisationParamRegressor(nn.Module):
    def __init__(self, add_coord=True, scale=None):
        # if scale is None, learn it. Otherwise it must be a float value for the scale
        super().__init__()
        self.scale = scale
        self.cnn_model = Sequential(
            nn.Conv2d(in_channels=5 if add_coord else 3, out_channels=16, kernel_size=5, stride=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.fc_model = Sequential(
            nn.Flatten(),
            nn.Linear(in_features=960, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=3) if scale is None else nn.Linear(in_features=64, out_features=2)
        )

        self.fc_model[-1].weight.data.zero_()
        if scale is None:
            self.fc_model[-1].bias.data.copy_(torch.tensor([1, 0, 0], dtype=torch.float))
        else:
            self.fc_model[-1].bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))

    def forward(self, x):
        """
        Returns attention affine transform
        [ s 0 t_x]
        [ 0 s t_y]
        """
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
        return out


class SpatialTransformerNetwork(MetaModule):
    def __init__(self, localisation_param_regressor, model, output_size,
                 sampling_type=STN_SamplingType.DEFAULT_BILINEAR, linearised_samples=8):
        super().__init__()
        self.localisation_param_regressor = localisation_param_regressor
        self.model = model
        # output_size (w_p, h_p)
        self.w, self.h = output_size
        self.sampling_type = sampling_type
        self.transformed_image = None
        self.transformation_params = None
        if sampling_type == STN_SamplingType.LINEARISED:
            # number of samples to take
            # the higher the number, the better the performance/gradients, but computationally more expensive
            LinearizedMutilSampler.num_grid = linearised_samples

    def forward(self, x, params=None):
        if isinstance(x, tuple):
            image_batch, _, _, _, _ = x
        else:
            image_batch = x
        b, c, h, w = image_batch.size()
        transformation_params = self.localisation_param_regressor(image_batch)
        # transformation_params (b, 2 * 3)
        transformation_params = transformation_params.view(b, 2, 3)
        grid = nn.functional.affine_grid(transformation_params, (b, c, self.h, self.w), align_corners=False)
        if self.sampling_type == STN_SamplingType.DEFAULT_BILINEAR:
            image_batch = nn.functional.grid_sample(image_batch, grid, align_corners=False)
        else:
            image_batch = LinearizedMutilSampler.linearized_grid_sample(image_batch, grid, "zeros")

        self.transformation_params = transformation_params
        # so we can access it, to plot
        self.transformed_image = image_batch
        return self.model(image_batch.contiguous(), params=params)
