import torch
from torch import nn

from lib.dsae.dsae import SpatialSoftArgmax, DSAE_Decoder, DSAE_Loss


# Networks

class TargetDecoder(nn.Module):
    """
    All Decoders inheriting from this class must return two values from their forward method:
    the reconstruction loss and the predicted action (usually tip velocity and rotation)
    """
    pass


class SoftTargetVectorDSAE_Decoder(TargetDecoder):
    def __init__(self, image_output_size, latent_dimension, normalise, encoder):
        # TODO: for the time being, assume only one attended feature -> there might be more
        super().__init__()
        self.default_decoder = DSAE_Decoder(
            image_output_size=image_output_size,
            latent_dimension=latent_dimension,
            normalise=normalise
        )
        # from a single point, predict the target
        # TODO: for multi attention to K points, in_features should be 2 * K
        self.fc1 = nn.Linear(in_features=2, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=6)
        self.activ = nn.ReLU()

        # Attention part
        self.encoder_ref = encoder
        # TODO: do not assume only one object attended to
        self.target_spatial = SpatialSoftArgmax(normalise=True)
        # one alpha per feature map
        self.att_param = nn.Parameter(torch.ones(latent_dimension // 2, 2))
        self.temperature = nn.Parameter(torch.ones(1))
        self.attended_location = None

    def forward(self, x):
        # TODO: minimise entropy too?
        b, _ = x.size()
        recon = self.default_decoder(x)
        visual_features = self.encoder_ref.visual_features
        b, c, h, w = visual_features.size()
        # spatial features (B, C*2) -> (B, C, 2)
        spatial_features = x.view(b, c, 2)
        # attention per point (C, 2)
        # dot product between spatial feature of dimension two with weights, gives (B, C)
        weighted_spatial_features = torch.sum(spatial_features * self.att_param, dim=-1)
        # attention weights (B, C)
        attention_weights = nn.functional.softmax(weighted_spatial_features / self.temperature, dim=1)
        # use attention weights on visual features to estimate a single feature map
        # (B, C, H * W)
        mult_visual_vectors = visual_features.view(b, c, h * w) * attention_weights.unsqueeze(-1)
        attended_visual_feature_map = torch.sum(mult_visual_vectors.view(b, c, h, w), dim=1, keepdim=True)
        # extract spatial location from this visual feature map of (B, 1, H, W) -> (B, 1, 2) and remove the 1
        self.attended_location = self.target_spatial(attended_visual_feature_map).squeeze(1)
        # use spatial point to predict target
        out_fc1 = self.activ(self.fc1(self.attended_location))
        target_vel_rot = self.fc2(out_fc1)
        return recon, target_vel_rot


class TargetVectorDSAE_Decoder(TargetDecoder):
    def __init__(self, image_output_size, latent_dimension, normalise):
        super().__init__()
        self.default_decoder = DSAE_Decoder(
            image_output_size=image_output_size,
            latent_dimension=latent_dimension,
            normalise=normalise
        )
        self.fc1 = nn.Linear(in_features=32, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=6)
        self.activ = nn.ReLU()

    def forward(self, x):
        b, _ = x.size()
        recon = self.default_decoder(x)
        out_fc1 = self.activ(self.fc1(x))
        target_vel_rot = self.fc2(out_fc1)
        return recon, target_vel_rot


class TargetVectorLoss(object):
    def __init__(self, add_g_slow):
        self.dsae_loss = DSAE_Loss(add_g_slow=add_g_slow)
        self.loss = nn.MSELoss(reduction="sum")

    def __call__(self, reconstructed, target, ft_minus1, ft, ft_plus1, pred_vel_rot, target_vel_rot):
        return (
            *self.dsae_loss(reconstructed, target, ft_minus1, ft, ft_plus1), self.loss(pred_vel_rot, target_vel_rot)
        )


class CustomDSAE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = (256, 128, 64, 32, 16)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_channels[0], kernel_size=7, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=5)
        self.batch_norm2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=5)
        self.batch_norm3 = nn.BatchNorm2d(out_channels[2])
        self.activ = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=out_channels[2], out_channels=out_channels[3], kernel_size=5)
        self.batch_norm4 = nn.BatchNorm2d(out_channels[3])
        self.conv5 = nn.Conv2d(in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=5)
        self.batch_norm5 = nn.BatchNorm2d(out_channels[4])
        self.spatial_soft_argmax = SpatialSoftArgmax(normalise=True)

    def forward(self, x):
        out_conv1 = self.activ(self.batch_norm1(self.conv1(x)))
        out_conv2 = self.activ(self.batch_norm2(self.conv2(out_conv1)))
        out_conv3 = self.activ(self.batch_norm3(self.conv3(out_conv2)))
        out_conv4 = self.activ(self.batch_norm4(self.conv4(out_conv3)))
        out_conv5 = self.activ(self.batch_norm5(self.conv5(out_conv4)))
        out = self.spatial_soft_argmax(out_conv5)
        return out


class CustomDSAE_Decoder(nn.Module):
    def __init__(self, image_output_size, latent_dimension):
        super().__init__()
        self.height, self.width = image_output_size
        self.latent_dimension = latent_dimension
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dimension, latent_dimension // 2, (3, 4)),
            nn.BatchNorm2d(latent_dimension // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dimension // 2, latent_dimension // 4, 4, 2, 1),
            nn.BatchNorm2d(latent_dimension // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dimension // 4, latent_dimension // 8, 4, 2, 1),
            nn.BatchNorm2d(latent_dimension // 8),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dimension // 8, latent_dimension // 16, 4, 2, 1),  # 24x32
            nn.BatchNorm2d(latent_dimension // 16),
            nn.ReLU(),
            nn.ConvTranspose2d(latent_dimension // 16, 1, 4, 2, 1),  # 48x64
            nn.Tanh()
        )

    def forward(self, x):
        out = self.decoder(x.unsqueeze(-1).unsqueeze(-1))
        b, c, h, w = out.size()
        assert h == self.height and w == self.width
        return out
