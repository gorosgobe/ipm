import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.dsae.dsae import CustomDeepSpatialAutoencoder, DSAE_Encoder
from lib.dsae.dsae_manager import DSAEManager
from lib.dsae.dsae_networks import TargetVectorDSAE_Decoder
from lib.dsae.dsae_test import DSAE_FeatureTest
from lib.common.utils import get_demonstrations, get_seed, set_up_cuda
from lib.dsae.dsae_dataset import DSAE_Dataset
from lib.dsae.dsae_discrim import SoftSpatialDiscriminator, DiscriminatorManager, DiscriminatorFeatureProvider

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", default="random")
    parser.add_argument("--disc_loss_params", nargs=2, type=float, required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--training", type=float, required=True)
    # set to 2 or 4
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
        latent_dimension=32,
        lr=0.001,
        num_epochs=parse_result.epochs,
        batch_size=128,
        split=[0.8, 0.1, 0.1],
        training=parse_result.training,
        disc_loss_params=parse_result.disc_loss_params,
        path=parse_result.path,
        output_divisor=4
    )

    height, width = config["size"]
    # transform for comparison between real and outputted image
    reduce_grayscale = transforms.Compose([
        transforms.Resize(size=(height // config["output_divisor"], width // config["output_divisor"])),
        transforms.Grayscale()
    ])

    dataset = DSAE_Dataset(
        root_dir=dataset_name,
        velocities_csv=f"{dataset_name}/velocities.csv",
        metadata=f"{dataset_name}/metadata.json",
        rotations_csv=f"{dataset_name}/rotations.csv",
        input_resize_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(height, width))
        ]),
        reduced_transform=reduce_grayscale,
        size=config["size"]
    )

    training_demonstrations, validation_demonstrations, test_demonstrations \
        = get_demonstrations(dataset, config["split"], limit_train_coeff=config["training"])

    dataloader = DataLoader(dataset=training_demonstrations, batch_size=config["batch_size"], shuffle=True,
                            num_workers=8)
    validation_dataloader = DataLoader(dataset=validation_demonstrations, batch_size=config["batch_size"], shuffle=True,
                                       num_workers=8)
    test_dataloader = DataLoader(dataset=test_demonstrations, batch_size=config["batch_size"], shuffle=False,
                                 num_workers=8)

    upsample_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config["size"]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat((x, x, x))),
    ])
    grayscale = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    print("Loading DSAE parameters...")
    state_dict = DSAEManager.load_state_dict(os.path.join("models/dsae/", config["path"]))
    model = CustomDeepSpatialAutoencoder(
        encoder=DSAE_Encoder(in_channels=3, out_channels=(64, 32, 16), strides=(2, 1, 1), normalise=True),
        decoder=TargetVectorDSAE_Decoder(
            image_output_size=(height // config["output_divisor"], width // config["output_divisor"]),
            latent_dimension=config["latent_dimension"],
            normalise=True
        )
    )
    model.load_state_dict(state_dict)
    model.to(config["device"])
    print("Loading done.")

    # once we have spatial features, train discriminator
    discriminator = SoftSpatialDiscriminator(latent_dimension=config["latent_dimension"]).to(config["device"])
    feature_provider = DiscriminatorFeatureProvider(model=model)
    discriminator_manager = DiscriminatorManager(
        name=f"disc_{config['name']}",
        feature_provider=feature_provider,
        model=discriminator,
        num_epochs=config["num_epochs"],
        optimiser=torch.optim.Adam(discriminator.parameters(), lr=config["lr"]),
        loss_params=config["disc_loss_params"],  # (0.5, 1.0)
        device=config["device"],
        patience=10,
        plot_params=dict(
            dataset=training_demonstrations,
            upsample_transform=upsample_transform,
            grayscale=grayscale,
            latent_dimension=config["latent_dimension"],
            feature_model=model
        ),
        plot=False
    )
    # same data as autoencoder
    discriminator_manager.train(dataloader, validation_dataloader)
    discriminator_manager.save_best_model("models/dsae/disc")

    print("Testing features...")
    feature_tester = DSAE_FeatureTest(
        model=discriminator,
        size=config["size"],
        device=config["device"],
        feature_provider=feature_provider,
        discriminator_mode=True
    )
    l2_errors, l1_errors = feature_tester.test(test_dataloader=test_dataloader)
    print("Test results:")
    min_l2_error, min_l2_index = torch.min(l2_errors, 0)
    print(f"L2 {l2_errors}, min error {min_l2_error.item()} at feature {min_l2_index.item()}")
    min_l1_error, min_l1_index = torch.min(l1_errors, 0)
    print(f"L1 {l1_errors}, min error {min_l1_error.item()} at feature {min_l1_index.item()}")
