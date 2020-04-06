import argparse

import numpy as np
from torch.utils.data import DataLoader

from lib.cv.controller import TrainingPixelROI, CropDeviationSampler
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.networks import *
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.common.utils import get_preprocessing_transforms, set_up_cuda, get_demonstrations, get_loss, get_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--training", type=float)
    parser.add_argument("--dataset")
    parser.add_argument("--version")
    parser.add_argument("--size", type=int)
    parser.add_argument("--random_std", type=int)
    parser.add_argument("--loss")
    parser.add_argument("--seed")
    parse_result = parser.parse_args()

    loss_params = get_loss(parse_result.loss)
    seed = get_seed(parse_result.seed)
    print("Seed:", seed)

    version = parse_result.version or "V1"

    size = (64, 48)
    divisor = 2
    if parse_result.size == 32:
        size = (32, 24)
        divisor = 4
    print("Cropped image size:", size)
    print("Training cropper divisor:", divisor)
    print("Attention network version:", version)

    add_spatial_maps = False
    if version == "V1":
        version = AttentionNetwork_32 if parse_result.size == 32 else AttentionNetwork
    elif version == "V2":
        version = AttentionNetworkV2_32 if parse_result.size == 32 else AttentionNetworkV2
    elif version.lower() == "coord":
        version = AttentionNetworkCoord_32 if parse_result.size == 32 else AttentionNetworkCoord
        add_spatial_maps = True
    elif version.lower() == "tile":
        version = AttentionNetworkTile_32 if parse_result.size == 32 else AttentionNetworkTile
    else:
        raise ValueError(f"Attention network version {version} is not available")

    dataset = parse_result.dataset or "text_camera_rand"
    print("Dataset: ", dataset)

    # random crop, if required
    crop_deviation_sampler = None
    if parse_result.random_std is not None:
        std = parse_result.random_std
        print(f"Random cropping, std {std}")
        crop_deviation_sampler = CropDeviationSampler(std=std)

    config = dict(
        seed=seed,
        # if pixel cropper is used to decrease size by two in both directions, size has to be decreased accordingly
        # otherwise we would be feeding a higher resolution cropped image
        # we want to supply a cropped image, corresponding exactly to the resolution of that area in the full image
        size=size,
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        initial_pixel_cropper=TrainingPixelROI(
            480 // divisor, 640 // divisor, add_spatial_maps=add_spatial_maps,
            crop_deviation_sampler=crop_deviation_sampler
        ),
        cache_images=False,
        batch_size=32,
        split=[0.8, 0.1, 0.1],
        name=parse_result.name or "AttentionNetworkRand",
        learning_rate=0.0001,
        max_epochs=250,
        validate_epochs=1,
        save_to_location="models/",
        network_klass=version,
        loss_params=loss_params
    )

    print("Name:", config["name"])

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = set_up_cuda(config["seed"])
    preprocessing_transforms, transforms = get_preprocessing_transforms(config["size"], is_coord=add_spatial_maps)

    dataset = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        initial_pixel_cropper=config["initial_pixel_cropper"],
        transform=preprocessing_transforms,
    )

    limit_training_coefficient = parse_result.training or 0.8  # all training data
    print("Training coeff limit:", limit_training_coefficient)

    training_demonstrations, val_demonstrations, test_demonstrations = get_demonstrations(dataset, config["split"],
                                                                                          limit_training_coefficient)

    train_data_loader = DataLoader(training_demonstrations, batch_size=config["batch_size"], num_workers=8,
                                   shuffle=True)
    validation_data_loader = DataLoader(val_demonstrations, batch_size=32, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_demonstrations, batch_size=32, num_workers=8, shuffle=True)

    tip_velocity_estimator = TipVelocityEstimator(
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        image_size=config["size"],
        network_klass=config["network_klass"],
        # transforms without initial resize, so they can be pickled correctly
        transforms=transforms,
        name=config["name"],
        device=device,
        composite_loss_params=config["loss_params"]
    )

    tip_velocity_estimator.train(
        data_loader=train_data_loader,
        max_epochs=config["max_epochs"],  # or stop early with patience 10
        validate_epochs=config["validate_epochs"],
        val_loader=validation_data_loader,
        test_loader=test_data_loader
    )

    # save_best_model
    tip_velocity_estimator.save_best_model(config["save_to_location"])
    # tip_velocity_estimator.plot_train_val_losses()
