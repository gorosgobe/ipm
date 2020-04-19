import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dsae.dsae_discrim import SoftSpatialDiscriminator, DiscriminatorManager, DiscriminatorFeatureProvider
from lib.dsae.dsae_manager import DSAEManager
from lib.common.utils import get_seed, set_up_cuda, get_demonstrations
from lib.dsae.dsae import DSAE_Loss, CustomDeepSpatialAutoencoder, DSAE_Encoder
from lib.dsae.dsae import DeepSpatialAutoencoder
from lib.dsae.dsae_networks import TargetVectorDSAE_Decoder, TargetVectorLoss, SoftSpatialTargetVectorDSAE_Decoder, \
    SoftVisualTargetVectorDSAE_Decoder
from lib.dsae.dsae_dataset import DSAE_Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", default="random")
    parser.add_argument("--g_slow", required=True)
    parser.add_argument("--loss_params", nargs=3, type=float, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--training", type=float, required=True)
    # set to 2 or 4
    parser.add_argument("--output_divisor", type=int, required=True)
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
        add_g_slow=parse_result.g_slow == "yes",
        version=parse_result.version,
        output_divisor=parse_result.output_divisor,
        split=[0.8, 0.1, 0.1],
        training=parse_result.training,
        criterion_params=parse_result.loss_params
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

    dataloader = DataLoader(dataset=training_demonstrations, batch_size=config["batch_size"], shuffle=True, num_workers=8)
    validation_dataloader = DataLoader(dataset=validation_demonstrations, batch_size=config["batch_size"], shuffle=True, num_workers=8)

    if config["version"] == "mse":
        model = DeepSpatialAutoencoder(
            in_channels=3,
            out_channels=(64, 32, 16),
            latent_dimension=config["latent_dimension"],
            # in the paper they output a reconstructed image 4 times smaller
            image_output_size=(height // config["output_divisor"], width // config["output_divisor"]),
            normalise=True
        )
    elif config["version"] == "target":
        model = CustomDeepSpatialAutoencoder(
            encoder=DSAE_Encoder(in_channels=3, out_channels=(64, 32, 16), strides=(2, 1, 1), normalise=True),
            decoder=TargetVectorDSAE_Decoder(
                image_output_size=(height // config["output_divisor"], width // config["output_divisor"]),
                latent_dimension=config["latent_dimension"],
                normalise=True
            )
        )
    else:
        # TODO: choose type of decoder here
        encoder = DSAE_Encoder(in_channels=3, out_channels=(64, 32, 16), strides=(2, 2, 1), normalise=True)
        model = CustomDeepSpatialAutoencoder(
            encoder=encoder,
            decoder=SoftVisualTargetVectorDSAE_Decoder(
                image_output_size=(height // config["output_divisor"], width // config["output_divisor"]),
                latent_dimension=config["latent_dimension"],
                normalise=True,
                encoder=encoder
            )
        )

    optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model = model.to(config["device"])

    if config["version"] == "mse":
        criterion = DSAE_Loss(add_g_slow=config["add_g_slow"])
    else:
        criterion = TargetVectorLoss(add_g_slow=config["add_g_slow"])

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

    trainer = DSAEManager(
        name=config["name"],
        model=model,
        num_epochs=config["num_epochs"],
        optimiser=optimiser,
        device=config["device"],
        criterion=criterion,
        criterion_params=config["criterion_params"],
        add_g_slow=config["add_g_slow"],
        patience=10,
        plot=True,
        plot_params=dict(
            dataset=training_demonstrations,
            upsample_transform=upsample_transform,
            grayscale=grayscale,
            latent_dimension=config["latent_dimension"]
        )
    )

    trainer.train(dataloader, validation_dataloader)
    trainer.save_best_model("models/dsae/")
    print("-" * 10)
    print("Autoencoder done, training discriminator...")
    print("-" * 10)
    # once we have spatial features, train discriminator
    discriminator = SoftSpatialDiscriminator(latent_dimension=config["latent_dimension"]).to(config["device"])
    discriminator_trainer = DiscriminatorManager(
        name=f"disc_{config['name']}",
        feature_provider=DiscriminatorFeatureProvider(model=model),
        model=discriminator,
        num_epochs=config["num_epochs"],
        optimiser=torch.optim.Adam(discriminator.parameters(), lr=config["lr"]),
        loss_params=(0.5, 1.0),
        device=config["device"],
        patience=10,
        plot_params=dict(
            dataset=training_demonstrations,
            upsample_transform=upsample_transform,
            grayscale=grayscale,
            latent_dimension=config["latent_dimension"],
            feature_model=model
        )
    )
    # same data as autoencoder
    discriminator_trainer.train(dataloader, validation_dataloader)
    discriminator_trainer.save_best_model("models/dsae/disc")
