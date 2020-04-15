import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import draw
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.common.utils import get_seed, set_up_cuda, get_preprocessing_transforms
from lib.dsae.dsae import DSAE_Loss
from lib.dsae.dsae import DeepSpatialAutoencoder
from lib.dsae.dsae_misc import DSAE_Dataset


def plot_images(epoch, name, model, upsample_transform, grayscale, device):
    f, axarr = plt.subplots(1, 2)
    model.eval()
    with torch.no_grad():
        sample = dataset[0]
        # get image and reconstruction in [0, 1] range
        image = sample["images"][0]
        reconstruction = (
                (model(image.to(device).unsqueeze(0)) + 1) / 2
        ).cpu()
        u_r_image = upsample_transform(reconstruction.squeeze(0)).numpy().transpose(1, 2, 0)

        # get spatial features (C, 2)
        features = model.encoder(image.to(device).unsqueeze(0)).squeeze(0).cpu()

        # normalise to 0, 1 (for drawing)
        image = (image + 1) / 2
        numpy_g_image = grayscale(image).numpy().transpose(1, 2, 0)
        # draw spatial features on image
        idx_features = len(features) - 1
        for idx, pos in enumerate(features):
            x, y = pos
            # x, y are in [-1, 1]
            x_pix = int((x + 1) * (128 - 1) / 2)
            y_pix = int((y + 1) * (96 - 1) / 2)
            rr, cc = draw.circle(x_pix, y_pix, radius=1.5, shape=numpy_g_image.shape)
            numpy_g_image[rr, cc] = np.array([1.0, 0.0, 0.0]) * (1 - idx / idx_features) + np.array(
                [0.0, 1.0, 0.0]) * idx / idx_features
        axarr[0].imshow(numpy_g_image)
        axarr[1].imshow(u_r_image)

    plt.savefig(f"{name}_{epoch}.png")
    plt.close()
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
    dataset_name = parse_result.dataset

    config = dict(
        seed=seed,
        name=parse_result.name,
        dataset_name=dataset_name,
        device=device,
        size=(96, 128),
        lr=0.001,
        num_epochs=parse_result.epochs,
        batch_size=128
    )

    height, width = config["size"]
    _, preprocessing_without_resize = get_preprocessing_transforms((width, height))
    # transform for comparison between real and outputted image
    reduce_grayscale = transforms.Compose([
        transforms.Resize(size=(height // 4, width // 4)),
        transforms.Grayscale()
    ])

    dataset = DSAE_Dataset(
        root_dir=dataset_name,
        velocities_csv=f"{dataset_name}/velocities.csv",
        metadata=f"{dataset_name}/metadata.json",
        input_resize_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(height, width))
        ]),
        reduced_transform=reduce_grayscale,
        normalising_transform=preprocessing_without_resize
    )

    dataloader = DataLoader(dataset=dataset, batch_size=config["batch_size"], shuffle=True, num_workers=16)

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

            # ft_minus1 = model.encoder(images[batch_size_range, 0, :])
            # print(ft_minus1.size())
            # ft = model.encoder(images[batch_size_range, 1, :])
            # print(ft.size())
            # ft_plus1 = model.encoder(images[batch_size_range, 2, :])
            # print(ft_plus1.size())
            loss = criterion(reconstructed=reconstructed, target=targets)
            # loss = criterion(reconstructed=reconstructed, target=targets, ft_minus1=ft_minus1, ft=ft,
            #                  ft_plus1=ft_plus1)
            loss_epoch += loss.item()

            loss.backward()
            optimiser.step()

        print(f"Epoch {epoch + 1}: {loss_epoch / len(dataloader.dataset)}")
        plot_images(epoch, config["name"], model, upsample_transform, grayscale, device)

    torch.save(model.state_dict(), f"{config['name']}.pt")
