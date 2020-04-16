import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from lib.cv.dataset import ImagesOnlyDataset
from lib.dsae.dsae import CoordinateUtils, SpatialSoftArgmax


class DSAE_Dataset(Dataset):
    def __init__(self, velocities_csv, metadata, root_dir, reduced_transform, input_resize_transform, size):
        # dataset that loads three successor images
        # for idx t, returns t-1, t, t+1
        # if at the boundary of demonstration, return t=0, 1, 2 or t=n-2, n-1, n
        # so images for t=0 are same for t=1; images for t=n-1 are the same as t=n
        self.images_dataset = ImagesOnlyDataset(
            velocities_csv=velocities_csv,
            metadata=metadata,
            root_dir=root_dir,
            as_uint=True
        )
        self.input_resize_transform = input_resize_transform
        self.reduced_transform = reduced_transform
        self.h, self.w = size

    def __len__(self):
        return self.images_dataset.__len__()

    def __getitem__(self, idx):
        d_data = self.images_dataset.get_demonstration_metadata(idx)
        start_idx = d_data["start"]
        end_idx = d_data["end"]
        center = 1

        if idx == start_idx:
            # idx, idx + 1, idx + 2
            idx += 1
            center = 0

        if idx == end_idx:
            # idx _2, idx - 1, idx
            idx -= 1
            center = 2

        # resize to input size
        prev_img = self.input_resize_transform(self.images_dataset[idx - 1])
        center_img = self.input_resize_transform(self.images_dataset[idx])
        next_img = self.input_resize_transform(self.images_dataset[idx + 1])
        imgs = (prev_img, center_img, next_img)

        # add coord maps if necessary
        normalising_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # get grayscaled output target image
        grayscaled = self.reduced_transform(imgs[center])  # resize to reduced size and grayscale
        sample = dict(
            images=torch.stack(list(map(lambda i: normalising_transform(i), imgs)), dim=0),
            center=center,
            target=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])(grayscaled)
        )
        return sample


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
        """
        Creates a Deep Spatial Autoencoder decoder
        :param image_output_size: (height, width) of the output, grayscale image
        :param latent_dimension: dimension of the low-dimensional encoded features.
        """
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
