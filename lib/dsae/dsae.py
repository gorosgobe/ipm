"""
This module implements, using PyTorch, the deep spatial autoencoder architecture presented in [1].
References:
    [1]: "Deep Spatial Autoencoders for Visuomotor Learning"
    Chelsea Finn, Xin Yu Tan, Yan Duan, Trevor Darrell, Sergey Levine, Pieter Abbeel
    Available at: https://arxiv.org/pdf/1509.06113.pdf
    [2]: https://github.com/tensorflow/tensorflow/issues/6271
"""

import torch
from torch import nn


class SpatialSoftArgmax(nn.Module):
    def __init__(self, temperature=None, normalise=False):
        """
        Applies a spatial soft argmax over the input images.
        :param temperature: The temperature parameter (float). If None, it is learnt.
        :param normalise: Should spatial features be normalised to range [-1, 1]?
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1)) if temperature is None else torch.tensor([temperature])
        self.normalise = normalise

    def forward(self, x):
        """
        Applies Spatial SoftArgmax operation on the input batch of images x.
        :param x: batch of images, of size (N, C, H, W)
        :return: Spatial features (one point per channel), of size (N, C, 2)
        """
        n, c, h, w = x.size()
        spatial_softmax_per_map = nn.functional.softmax(x.view(n * c, h * w) / self.temperature, dim=1)
        spatial_softmax = spatial_softmax_per_map.view(n, c, h, w)

        # calculate image coordinate maps
        x_range = torch.arange(w, dtype=torch.float32)
        y_range = torch.arange(h, dtype=torch.float32)
        if self.normalise:
            x_range = (x_range / (w - 1)) * 2 - 1
            y_range = (y_range / (h - 1)) * 2 - 1
        image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
        image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
        # size (H, W, 2)
        image_coordinates = torch.cat((image_x.unsqueeze(-1), image_y.unsqueeze(-1)), dim=-1)
        # send to device
        image_coordinates = image_coordinates.to(device=x.device)

        # multiply coordinates by the softmax and sum over height and width, like in [2]
        expanded_spatial_softmax = spatial_softmax.unsqueeze(-1)
        image_coordinates = image_coordinates.unsqueeze(0)
        out = torch.sum(expanded_spatial_softmax * image_coordinates, dim=[2, 3])
        # (N, C, 2)
        return out


class DSAE_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, temperature=None, normalise=False):
        """
        Creates a Deep Spatial Autoencoder encoder
        :param in_channels: Input channels in the input image
        :param out_channels: Output channels for each of the layers. The last output channel corresponds to half the
        size of the low-dimensional latent representation.
        :param temperature: Temperature for spatial soft argmax operation. See SpatialSoftArgmax.
        :param normalise: Normalisation of spatial features. See SpatialSoftArgmax.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=7, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels=out_channels[0], out_channels=out_channels[1], kernel_size=5)
        self.batch_norm2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=5)
        self.batch_norm3 = nn.BatchNorm2d(out_channels[2])
        self.activ = nn.ReLU()
        self.spatial_soft_argmax = SpatialSoftArgmax(temperature=temperature, normalise=normalise)

    def forward(self, x):
        out_conv1 = self.activ(self.batch_norm1(self.conv1(x)))
        out_conv2 = self.activ(self.batch_norm2(self.conv2(out_conv1)))
        out_conv3 = self.activ(self.batch_norm3(self.conv3(out_conv2)))
        out = self.spatial_soft_argmax(out_conv3)
        return out


class DSAE_Decoder(nn.Module):
    def __init__(self, image_output_size, latent_dimension):
        """
        Creates a Deep Spatial Autoencoder decoder
        :param image_output_size: (height, width) of the output, grayscale image
        :param latent_dimension: dimension of the low-dimensional encoded features.
        """
        super().__init__()
        self.height, self.width = image_output_size
        self.latent_dimension = latent_dimension
        self.decoder = nn.Linear(in_features=latent_dimension, out_features=self.height * self.width)

    def forward(self, x):
        out = self.decoder(x)
        out = out.view(-1, 1, self.height, self.width)
        return out


class DeepSpatialAutoencoder(nn.Module):
    def __init__(self, image_output_size, in_channels=3, out_channels=(64, 32, 16), latent_dimension=32,
                 temperature=None, normalise=False):
        """
        Creates a deep spatial autoencoder. Default parameters are the ones used in [1]. See docs for encoder and decoder.
        :param image_output_size: Reconstructed image size
        :param in_channels: Number of channels of input image
        :param out_channels: Output channels of each conv layer in the encoder.
        :param latent_dimension: Input dimension for decoder
        :param temperature: Temperature parameter, None if it is to be learnt
        :param normalise: Should spatial features be normalised to [-1, 1]?
        """
        super().__init__()
        if out_channels[-1] * 2 != latent_dimension:
            raise ValueError("Spatial SoftArgmax produces a location (x,y) per feature map!")
        self.encoder = DSAE_Encoder(in_channels=in_channels, out_channels=out_channels, temperature=temperature,
                                    normalise=normalise)
        self.decoder = DSAE_Decoder(image_output_size=image_output_size, latent_dimension=latent_dimension)

    def forward(self, x):
        # (N, C, 2)
        spatial_features = self.encoder(x)
        n, c, _2 = spatial_features.size()
        # (N, C * 2 = latent dimension)
        return self.decoder(spatial_features.view(n, c * 2))


class DSAE_Loss(object):
    def __init__(self, add_g_slow=True):
        """
        Loss for deep spatial autoencoder.
        :param add_g_slow: Should g_slow contribution be added? See [1].
        """
        self.add_g_slow = add_g_slow
        self.mse_loss = nn.MSELoss(reduction="sum")

    def __call__(self, reconstructed, target, ft_minus1=None, ft=None, ft_plus1=None):
        loss = self.mse_loss(reconstructed, target)
        if self.add_g_slow:
            loss += self.mse_loss(ft_plus1 - ft, ft - ft_minus1)
        return torch.mean(loss, dim=0)