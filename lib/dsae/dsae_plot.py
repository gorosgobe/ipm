import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import draw

from lib.dsae.dsae_networks import TargetDecoder


def plot_reconstruction_images(epoch, name, dataset, model, upsample_transform, grayscale, device):
    f, axarr = plt.subplots(5, 2, figsize=(10, 15), dpi=100)
    plt.tight_layout()
    model.eval()
    with torch.no_grad():
        for i in range(5):
            sample = dataset[34 + i * 4]
            # get image and reconstruction in [0, 1] range
            image = sample["images"][1]
            if isinstance(model.decoder, TargetDecoder):
                out_image, _ = model(image.to(device).unsqueeze(0))
            else:
                out_image = model(image.to(device).unsqueeze(0))

            reconstruction = (
                    (out_image + 1) * 255 / 2
            ).type(torch.uint8).cpu()
            u_r_image = upsample_transform(reconstruction.squeeze(0)).numpy().transpose(1, 2, 0)

            # get spatial features (C, 2)
            features = model.encoder(image.to(device).unsqueeze(0)).squeeze(0).cpu()

            # normalise to 0, 255 (for PIL, ToTensor then turns it into 0, 1)
            image = (image[:3, :, :] + 1) * 255 / 2
            numpy_g_image = grayscale(image.type(torch.uint8)).numpy().transpose(1, 2, 0)
            # draw spatial features on image
            idx_features = len(features) - 1
            for idx, pos in enumerate(features):
                x, y = pos
                # x, y are in [-1, 1]
                x_pix = int((x + 1) * (128 - 1) / 2)
                y_pix = int((y + 1) * (96 - 1) / 2)
                rr, cc = draw.circle(y_pix, x_pix, radius=2, shape=numpy_g_image.shape)
                numpy_g_image[rr, cc] = np.array([1.0, 0.0, 0.0]) * (1 - idx / idx_features) + np.array(
                    [0.0, 1.0, 0.0]) * idx / idx_features
            axarr[i, 0].imshow(numpy_g_image)
            axarr[i, 1].imshow(u_r_image)

    plt.savefig(f"{name}_{epoch}.png")
    plt.close()
    model.train()


def plot_full_demonstration(epoch, name, dataset, model, grayscale, device, latent_dim, rows=4):
    # track all features across demonstration, showing the first image only
    f, axarr = plt.subplots(nrows=rows, ncols=(latent_dim // (2 * rows)), figsize=(10, 10), dpi=200)
    plt.tight_layout()
    model.eval()
    feature_positions = {i: [] for i in range(latent_dim // 2)}
    with torch.no_grad():
        image = dataset[34]["images"][1]  # initial image for first demonstration
        # normalise to 0, 255 (for PIL, ToTensor then turns it into 0, 1)
        image = (image[:3, :, :] + 1) * 255 / 2
        numpy_g_image = grayscale(image.type(torch.uint8)).numpy().transpose(1, 2, 0)

        # collect features across trajectories
        for i in range(30):
            sample = dataset[34 + i]
            image = sample["images"][1]
            # get spatial features (C, 2)
            features = model.encoder(image.to(device).unsqueeze(0)).squeeze(0).cpu()

            for idx, pos in enumerate(features):
                x, y = pos
                # x, y are in [-1, 1]
                x_pix = int((x + 1) * (128 - 1) / 2)
                y_pix = int((y + 1) * (96 - 1) / 2)
                feature_positions[idx].append((x_pix, y_pix))

        # draw features across trajectory, interpolating colour across trajectory progress
        for i in feature_positions:
            image_to_draw_on = numpy_g_image.copy()
            for idx, feature in enumerate(feature_positions[i]):
                x_pix, y_pix = feature
                rr, cc = draw.circle(y_pix, x_pix, radius=2, shape=image_to_draw_on.shape)
                image_to_draw_on[rr, cc] = np.array([1.0, 0.0, 0.0]) * (1 - idx / len(feature_positions[i])) + \
                                           np.array([0.0, 1.0, 0.0]) * idx / len(feature_positions[i])
            axarr[i // 4, i % 4].imshow(image_to_draw_on)

    plt.savefig(f"demon_{name}_{epoch}.png")
    plt.close()
    model.train()
