import argparse
import os

from torch.utils.data import DataLoader

from lib.common.utils import get_preprocessing_transforms, set_up_cuda, get_demonstrations, get_seed
from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.meta.meta_networks import MetaAttentionNetworkCoord
from lib.networks import *
from lib.stn.stn import LocalisationParamRegressor, SpatialTransformerNetwork, STN_SamplingType
from lib.stn.stn_manager import STNManager
from lib.stn.stn_visualise import visualise

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--training", type=float)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--seed", default="random")
    parser.add_argument("--scale", type=float)
    parse_result = parser.parse_args()

    seed = get_seed(parse_result.seed)
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

    localisation_param_regressor = LocalisationParamRegressor(
        add_coord=True,
        scale=parse_result.scale
    )
    model = MetaAttentionNetworkCoord.create(*size)(track=True)
    add_spatial_maps = True

    stn = SpatialTransformerNetwork(
        localisation_param_regressor=localisation_param_regressor,
        model=model,
        output_size=size,
        sampling_type=STN_SamplingType.DEFAULT_BILINEAR
    )

    # size is 128, 96 for images, CoordConv takes downsampled version
    size = (128, 96)

    dataset = parse_result.dataset
    print("Dataset: ", dataset)

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
            480, 640, add_spatial_maps=add_spatial_maps,
        ),
        batch_size=32,
        split=[0.8, parse_result.val, 0.2 - parse_result.val],
        name=parse_result.name,
        max_epochs=parse_result.epochs,
        validate_epochs=1,
        save_to_location="models/meta_stn",
        patience=parse_result.patience,
    )

    print("Name:", config["name"])

    device = set_up_cuda(config["seed"])
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

    limit_training_coefficient = parse_result.training or 0.8  # all training data
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
        device=device
    )

    manager.train(num_epochs=config["max_epochs"], train_dataloader=train_data_loader,
                  val_dataloader=validation_data_loader, test_dataloader=test_data_loader)

    # save_best_model
    if config["max_epochs"] > 0:
        manager.save(os.path.join(config["save_to_location"], config["name"]))

    # visualise test images and their transformations
    if config["max_epochs"] > 0:
        stn.load_state_dict(manager.get_info()["stn_state_dict"])

    visualise(name=f"{config['name']}_train", model=stn, dataloader=train_data_loader)
    if config["split"][1] == 0.2:
        visualise(name=f"{config['name']}_val", model=stn, dataloader=validation_data_loader)
    else:
        visualise(name=f"{config['name']}_val", model=stn, dataloader=validation_data_loader)
        visualise(name=config["name"], model=stn, dataloader=test_data_loader)
