import argparse
import os

import torchvision
from torch.utils.data import DataLoader

from lib.common.utils import get_preprocessing_transforms, set_up_cuda, get_demonstrations, get_seed
from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.dsae.dsae import CustomDeepSpatialAutoencoder, DSAE_Encoder
from lib.dsae.dsae_manager import DSAEManager
from lib.dsae.dsae_networks import TargetVectorDSAE_Decoder
from lib.dsae.dsae_plot import plot_reconstruction_images
from lib.meta.meta_networks import MetaAttentionNetworkCoord
from lib.networks import *
from lib.stn.stn import LocalisationParamRegressor, SpatialTransformerNetwork, STN_SamplingType, \
    SpatialLocalisationRegressor
from lib.stn.stn_manager import STNManager
from lib.stn.stn_visualise import visualise

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("--limit", type=float, default=-1)
    parser.add_argument("--loc_lr", type=float, default=1e-4)
    parser.add_argument("--model_lr", type=float, default=1e-2)
    parser.add_argument("--retraining", type=float, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pretrain", default="yes")
    parser.add_argument("--seed", default="random")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--dsae_path")
    parse_result = parser.parse_args()

    seed = get_seed(parse_result.seed)
    device = set_up_cuda(seed)
    print("Seed:", seed)

    if parse_result.size == 32:
        size = (32, 24)
    elif parse_result.size == 64:
        size = (64, 48)
    elif parse_result.size == 128:
        size = (128, 96)
    else:
        raise ValueError("Invalid size!")

    print("Downscale transformed image to:", size)

    if parse_result.dsae_path is not None:
        dsae = CustomDeepSpatialAutoencoder(
            encoder=DSAE_Encoder(
                in_channels=3,
                out_channels=(256, 128, 64),
                strides=(2, 1, 1),
                normalise=True
            ),
            decoder=TargetVectorDSAE_Decoder(
                image_output_size=(32, 24),
                latent_dimension=128,
                normalise=True
            )
        )
        dsae.load_state_dict(DSAEManager.load_state_dict(os.path.join("models/dsae", parse_result.dsae_path)))
        dsae = dsae.to(device)
        localisation_param_regressor = SpatialLocalisationRegressor(
            dsae=dsae.encoder,
            latent_dimension=128,
            scale=parse_result.scale
        )
    else:
        localisation_param_regressor = LocalisationParamRegressor(
            add_coord=True,
            scale=parse_result.scale
        )

    add_spatial_maps = True
    model = MetaAttentionNetworkCoord.create(*size)(track=True)

    stn = SpatialTransformerNetwork(
        localisation_param_regressor=localisation_param_regressor,
        model=model,
        output_size=size,
        sampling_type=STN_SamplingType.LINEARISED
    )

    dataset = parse_result.dataset
    print("Dataset: ", dataset)

    config = dict(
        seed=seed,
        # size is 128, 96 for images, CoordConv takes downsampled version
        size=(128, 96),
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        initial_pixel_cropper=TrainingPixelROI(
            480, 640, add_spatial_maps=add_spatial_maps,
        ),
        batch_size=parse_result.batch_size,
        split=parse_result.split,
        name=parse_result.name,
        max_epochs=parse_result.epochs,
        validate_epochs=1,
        save_to_location="models/meta_stn",
        patience=parse_result.patience,
        pretrain=parse_result.pretrain == "yes",
        dsae_path=parse_result.dsae_path,
        device=device,
        retraining=parse_result.retraining,
    )

    print("Name:", config["name"])

    preprocessing_transforms, transforms = get_preprocessing_transforms(config["size"], is_coord=add_spatial_maps)

    dataset = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        initial_pixel_cropper=config["initial_pixel_cropper"],
        transform=preprocessing_transforms,
        ignore_cache_if_cropper=device == torch.device("cpu")
    )

    limit_training_coefficient = parse_result.limit  # all training data
    print("Training coeff limit:", limit_training_coefficient)

    training_demonstrations, val_demonstrations, test_demonstrations = get_demonstrations(dataset, config["split"],
                                                                                          limit_training_coefficient)

    train_data_loader = DataLoader(training_demonstrations, batch_size=config["batch_size"], num_workers=8,
                                   shuffle=True)
    validation_data_loader = DataLoader(val_demonstrations, batch_size=32, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_demonstrations, batch_size=32, num_workers=8, shuffle=False)

    manager = STNManager(
        name=config["name"],
        stn=stn,
        device=device,
        loc_lr=parse_result.loc_lr,
        model_lr=parse_result.model_lr,
    )

    manager.train(num_epochs=config["max_epochs"], train_dataloader=train_data_loader,
                  val_dataloader=validation_data_loader, test_dataloader=test_data_loader,
                  pre_training=config["pretrain"])

    # save_best_model
    if config["max_epochs"] > 0:
        manager.save_best_model(config["save_to_location"])

    # visualise test images and their transformations
    if config["max_epochs"] > 0:
        stn.load_state_dict(manager.get_best_info()["stn_state_dict"])

    visualise(name=f"{config['name']}_train", model=stn, dataloader=train_data_loader)
    visualise(name=f"{config['name']}_val", model=stn, dataloader=validation_data_loader)
    if config["split"][2] != 0.0:
        visualise(name=config["name"], model=stn, dataloader=test_data_loader)

    if parse_result.dsae_path is not None:
        upsample_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((96, 128)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x))),
        ])
        grayscale = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor()
        ])
        plot_reconstruction_images(
            epoch="train_dsae", name=config["name"], dataset=training_demonstrations, model=dsae,
            attender=None, upsample_transform=upsample_transform,
            grayscale=grayscale, device=config["device"]
        )
        plot_reconstruction_images(
            epoch="val_dsae", name=config["name"], dataset=val_demonstrations, model=dsae,
            attender=None, upsample_transform=upsample_transform,
            grayscale=grayscale, device=config["device"]
        )
        if config["split"][2] != 0.0:
            plot_reconstruction_images(
                epoch="test_dsae", name=config["name"], dataset=test_demonstrations, model=dsae,
                attender=None, upsample_transform=upsample_transform,
                grayscale=grayscale, device=config["device"]
            )

    print("Retraining...")
    training_demonstrations, val_demonstrations, test_demonstrations = get_demonstrations(dataset, [0.8, 0.1, 0.1],
                                                                                          config["retraining"])
    train_data_loader = DataLoader(training_demonstrations, batch_size=config["batch_size"], num_workers=8,
                                   shuffle=True)
    validation_data_loader = DataLoader(val_demonstrations, batch_size=32, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_demonstrations, batch_size=32, num_workers=8, shuffle=False)
    # reinit coord conv
    model = MetaAttentionNetworkCoord.create(*size)(track=True)
    stn.model = model
    # resend to GPU, as model was created on CPU
    stn = stn.to(config["device"])
    # train with all the data and localisation param regressor in eval mode
    manager.name = manager.name + "_retrain"
    manager.retrain(
        num_epochs=1,
        train_dataloader=train_data_loader,
        val_dataloader=validation_data_loader,
        test_dataloader=test_data_loader
    )
    manager.save_best_model(config["save_to_location"])
