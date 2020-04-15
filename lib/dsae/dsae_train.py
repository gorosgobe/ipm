import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.common.utils import get_seed, set_up_cuda, get_preprocessing_transforms
from lib.dsae.dsae import DeepSpatialAutoencoder
from lib.dsae.dsae_misc import DSAE_Dataset, DSAE_Loss
from skimage import draw


def plot_images(epoch, model, upsample_transform, grayscale, axarr):
    model.eval()
    with torch.no_grad():
        sample = dataset[0]
        # get image and reconstruction in [0, 1] range
        image = sample["images"][0]
        reconstruction = (model(image.unsqueeze(0)) + 1) / 2

        # get spatial features (C, 2)
        features = model.encoder(image.unsqueeze(0)).squeeze(0)

        # normalise to 0, 1 (for drawing)
        image = (image + 1) / 2
        g_image = grayscale(image)
        numpy_g_image = g_image.numpy().transpose(1, 2, 0)
        # draw spatial features on image
        for pos in features:
            x, y = pos
            # x, y are in [-1, 1]
            x_pix = int((x + 1) * (128 - 1) / 2)
            y_pix = int((y + 1) * (96 - 1) / 2)
            rr, cc = draw.circle(x_pix, y_pix, radius=2, shape=numpy_g_image.shape)
            numpy_g_image[rr, cc] = np.array([1.0, 0.0, 0.0])
        u_r_image = upsample_transform(reconstruction.squeeze(0))
        axarr[epoch, 0].imshow(numpy_g_image)
        axarr[epoch, 1].imshow(u_r_image.numpy().transpose(1, 2, 0))
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", default="random")
    parse_result = parser.parse_args()

    seed = get_seed(parse_result.seed)
    device = set_up_cuda(seed)
    dataset_name = f"../../{parse_result.dataset}"

    config = dict(
        seed=seed,
        name=parse_result.name,
        dataset_name=dataset_name,
        device=device,
        size=(96, 128),
        lr=0.001,
        num_epochs=parse_result.epochs,
        batch_size=64
    )

    height, width = config["size"]
    preprocessing, _ = get_preprocessing_transforms((width, height))
    # transform for comparison between real and outputted image
    reduce_grayscale = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(height // 4, width // 4)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    dataset = DSAE_Dataset(
        root_dir=dataset_name,
        velocities_csv=f"{dataset_name}/velocities.csv",
        metadata=f"{dataset_name}/metadata.json",
        reduced_transform=reduce_grayscale,
        transform=preprocessing
    )

    dataloader = DataLoader(dataset=dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)

    model = DeepSpatialAutoencoder(
        in_channels=3,
        out_channels=(64, 32, 16),
        latent_dimension=32,
        # in the paper they output a reconstructed image 4 times smaller
        image_output_size=(height // 4, width // 4),
        normalise=True
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model = model.to(config["device"])
    model.train()

    criterion = DSAE_Loss(add_g_slow=False)

    upsample_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat((x, x, x))),
    ])
    grayscale = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    f, axarr = plt.subplots(config["num_epochs"], 2)

    for epoch in range(config["num_epochs"]):
        loss_epoch = 0
        for batch_idx, batch in enumerate(dataloader):
            optimiser.zero_grad()
            images = batch["images"].to(device)
            centers = batch["center"].to(device)
            targets = batch["target"].to(device)
            # select image to reconstruct
            batch_size_range = torch.arange(images.size()[0])
            center_images = images[batch_size_range, centers, :]
            reconstructed = model(center_images)
            loss = criterion(reconstructed=reconstructed, target=targets, ft_minus1=None, ft=None, ft_plus1=None)
            loss_epoch += loss.item()

            loss.backward()
            optimiser.step()

        print(f"Epoch {epoch + 1}: {loss_epoch / len(dataloader)}")
        plot_images(epoch, model, upsample_transform, grayscale, axarr)

    plt.savefig(f"{config['name']}.png")

