import argparse

import numpy as np
from torch.utils.data import DataLoader

from lib.cv.dataset import BaselineTipVelocitiesDataset
from lib.networks import *
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.common.utils import set_up_cuda, get_demonstrations, get_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--training", type=float)
    parser.add_argument("--dataset")
    parser.add_argument("--version")
    parser.add_argument("--seed")
    parse_result = parser.parse_args()

    seed = get_seed(parse_result.seed)
    print("Seed:", seed)
    dataset = parse_result.dataset
    print("Dataset: ", dataset)
    version = parse_result.version or "default"
    if version != "default" and version != "sim_params_coord":
        raise ValueError("Incorrect version for baseline network")
    print("Version: ", version)

    config = dict(
        seed=seed,
        size=(128, 96),  # ignored anyway
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        batch_size=32,
        split=[0.8, 0.1, 0.1],
        name=parse_result.name or "BaselineNetworkRand",
        learning_rate=0.001,
        max_epochs=500,
        validate_epochs=1,
        save_to_location="models/",
        network_klass=BaselineSimilarParamsAttentionCoord64 if version == "sim_params_coord" else BaselineNetwork
    )

    print("Name:", config["name"])

    device = set_up_cuda(config["seed"])

    dataset = BaselineTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"]
    )

    limit_training_coefficient = parse_result.training or 0.8  # all training data
    print("Training coeff limit:", limit_training_coefficient)
    training_demonstrations, val_demonstrations, test_demonstrations = get_demonstrations(dataset, config["split"], limit_training_coefficient)

    train_data_loader = DataLoader(training_demonstrations, batch_size=config["batch_size"], num_workers=8,
                                   shuffle=True)
    validation_data_loader = DataLoader(val_demonstrations, batch_size=64, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_demonstrations, batch_size=64, num_workers=8, shuffle=True)

    tip_velocity_estimator = TipVelocityEstimator(
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        image_size=config["size"],
        network_klass=config["network_klass"],
        # transforms without initial resize, so they can be pickled correctly
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
    #tip_velocity_estimator.plot_train_val_losses()
