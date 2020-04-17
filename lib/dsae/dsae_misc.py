import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from dataset import ImageTipVelocitiesDataset
from lib.dsae.dsae import SpatialSoftArgmax, DSAE_Decoder, DSAE_Loss


class DSAE_Dataset(ImageTipVelocitiesDataset):
    def __init__(self, velocities_csv, rotations_csv, metadata, root_dir, reduced_transform, input_resize_transform,
                 size):
        # dataset that loads three successor images
        # if at boundary, load the boundary image twice
        super().__init__(
            velocities_csv=velocities_csv,
            rotations_csv=rotations_csv,
            metadata=metadata,
            root_dir=root_dir,
            as_uint=True
        )
        self.input_resize_transform = input_resize_transform
        self.reduced_transform = reduced_transform
        self.h, self.w = size

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        d_data = super().get_demonstration_metadata(idx)
        start_idx = d_data["start"]
        end_idx = d_data["end"]
        idx_prev = idx - 1
        idx_next = idx + 1

        if idx == start_idx:
            # idx, idx, idx + 1
            idx_prev = idx

        if idx == end_idx:
            # idx - 1, idx, idx
            idx_next = idx

        # resize to input size
        prev_img = self.input_resize_transform(super().__getitem__(idx_prev)["image"])

        curr_sample = super().__getitem__(idx)
        center_img = self.input_resize_transform(curr_sample["image"])
        center_target_vel_rot = torch.cat((
            torch.tensor(curr_sample["tip_velocities"]), torch.tensor(curr_sample["rotations"])
        ))
        next_img = self.input_resize_transform(super().__getitem__(idx_next)["image"])

        imgs = (prev_img, center_img, next_img)

        normalising_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # get grayscaled output target image
        grayscaled = self.reduced_transform(imgs[1])  # resize to reduced size and grayscale
        sample = dict(
            images=torch.stack(list(map(lambda i: normalising_transform(i), imgs)), dim=0),
            target_image=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])(grayscaled),
            target_vel_rot=center_target_vel_rot
        )
        return sample


# Networks

class TargetDecoder(nn.Module):
    """
    All Decoders inheriting from this class must return two values from their forward method:
    the reconstruction loss and the predicted action (usually tip velocity and rotation)
    """
    pass


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
