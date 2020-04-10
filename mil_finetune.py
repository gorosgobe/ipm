import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.common.utils import get_demonstrations, get_seed, get_divisors, set_up_cuda, get_preprocessing_transforms
from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.networks import AttentionNetworkCoord_32, AttentionNetworkCoord
from lib.meta.mil import MetaImitationLearning

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--load_params_name", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", default="random")
    parser.add_argument("--dataset", required=True)
    parse_result = parser.parse_args()

    seed = get_seed(parse_result.seed)

    if parse_result.size == 32:
        cropped_size = (32, 24)
        network = AttentionNetworkCoord_32
    elif parse_result.size == 64:
        cropped_size = (64, 48)
        network = AttentionNetworkCoord
    else:
        raise ValueError("Invalid size argument, it has to be either 32 or 64")

    dataset_name = parse_result.dataset
    config = dict(
        name=parse_result.name,
        load_params_name=parse_result.load_params_name,
        size=(128, 96),
        cropped_size=cropped_size,
        batch_size=32,
        network=network,
        validate_epochs=1,
        max_epochs=250,
        seed=seed,
        dataset=dataset_name,
        velocities_csv=f"{dataset_name}/velocities.csv",
        rotations_csv=f"{dataset_name}/rotations.csv",
        metadata=f"{dataset_name}/metadata.json",
        root_dir=dataset_name
    )

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = set_up_cuda(seed=get_seed(parsed_seed=config["seed"]))
    preprocessing_transforms, transforms = get_preprocessing_transforms(size=config["cropped_size"], is_coord=True)

    # Once trained, load parameters and train with those as the initialisation parameters
    # use the best initialisation parameters in terms of the validation outer loss
    parameter_state_dict = MetaImitationLearning.load_best_params(f"models/pretraining_test/{config['load_params_name']}")
    true_model = config["network"](-1, -1)
    true_model.load_state_dict(parameter_state_dict, strict=False)

    divisor_width, divisor_height = get_divisors(*config["size"], *config["cropped_size"])
    dataset = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        initial_pixel_cropper=TrainingPixelROI(480 // divisor_height, 640 // divisor_width, add_spatial_maps=True),
        transform=preprocessing_transforms,
    )

    training_demonstrations, val_demonstrations, test_demonstrations = get_demonstrations(dataset, config["split"])

    train_data_loader = DataLoader(training_demonstrations, batch_size=config["batch_size"], num_workers=8,
                                   shuffle=True)
    validation_data_loader = DataLoader(val_demonstrations, batch_size=32, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_demonstrations, batch_size=32, num_workers=8, shuffle=True)

    tip_velocity_estimator = TipVelocityEstimator(
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        image_size=config["size"],
        network=true_model,
        # transforms without initial resize, so they can be pickled correctly
        transforms=transforms,
        name=config["name"],
        device=device
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