import argparse
import pprint

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmeta.utils.data import BatchMetaDataLoader

from lib.common.utils import set_up_cuda, get_seed, get_preprocessing_transforms, get_demonstrations
from lib.cv.controller import RandomPixelROI, TrainingPixelROI
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.meta.meta_dataset import MILTipVelocityDataset, DatasetType
from lib.meta.meta_networks import MetaAttentionNetworkCoord, MetaAttentionNetworkCoord_32
from lib.meta.mil import MetaImitationLearning, MetaAlgorithm
from lib.networks import AttentionNetworkCoord_32
from lib.cv.tip_velocity_estimator import TipVelocityEstimator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mil_epochs", type=int, required=True)
    parse_result = parser.parse_args()

    seed = get_seed(parse_result.seed)

    if parse_result.size == 32:
        cropped_size = (32, 24)
        network = MetaAttentionNetworkCoord_32
    elif parse_result.size == 64:
        cropped_size = (64, 48)
        network = MetaAttentionNetworkCoord
    else:
        raise ValueError("Invalid size argument, it has to be either 32 or 64")

    dataset_name = parse_result.dataset
    config = dict(
        name=parse_result.name,
        size=(128, 96),
        cropped_size=(32, 24),
        seed=seed,
        save_to_location="models/pretraining_test",
        mil_epochs=parse_result.mil_epochs,
        mil_meta_algorithm=MetaAlgorithm.FOMAML,
        mil_adaptation_steps=1,
        mil_step_size=0.1,
        mil_learning_rate=0.01,
        mil_max_batches=100,  # if multiplied by batch size of meta data loader -> number of tasks per epoch
        optimiser=torch.optim.Adam,
        learning_rate=0.0001,
        loss_function=torch.nn.MSELoss(),
        split=[0.8, 0.1, 0.1],
        batch_size=32,
        validate_epochs=1,
        max_epochs=250,
        dataset=dataset_name,
        velocities_csv=f"{dataset_name}/velocities.csv",
        rotations_csv=f"{dataset_name}/rotations.csv",
        metadata=f"{dataset_name}/metadata.json",
        root_dir=dataset_name
    )
    print("Config:")
    pprint.pprint(config)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = set_up_cuda(seed=get_seed(parsed_seed=config["seed"]))
    preprocessing_transforms, transforms = get_preprocessing_transforms(size=config["cropped_size"], is_coord=True)

    model = network()
    mil = MetaImitationLearning(
        name=config["name"],
        model=model,
        meta_algorithm=config["mil_meta_algorithm"],
        num_adaptation_steps=config["mil_adaptation_steps"],
        step_size=config["mil_step_size"],
        optimiser=config["optimiser"](model.parameters(), lr=config["mil_learning_rate"]),
        loss_function=config["loss_function"],
        max_batches=config["mil_max_batches"],
        device=device
    )

    width, height = config["size"]
    cropped_width, cropped_height = config["cropped_size"]
    divisor_width = int(width / cropped_width)
    divisor_height = int(height / cropped_height)
    mil_dataset = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        transform=preprocessing_transforms,
        initial_pixel_cropper=RandomPixelROI(480 // divisor_height, 640 // divisor_width, add_spatial_maps=True),
    )

    meta_train_dataset = MILTipVelocityDataset(
        demonstration_dataset=mil_dataset,
        split=config["split"],
        dataset_type=DatasetType.TRAIN
    )

    meta_val_dataset = MILTipVelocityDataset(
        demonstration_dataset=mil_dataset,
        split=config["split"],
        dataset_type=DatasetType.VAL
    )

    train_batch_dataloader = BatchMetaDataLoader(meta_train_dataset, batch_size=1,
                                                 num_workers=8)
    val_batch_dataloader = BatchMetaDataLoader(meta_val_dataset, batch_size=1, num_workers=8)

    mil.train(
        train_batch_dataloader=train_batch_dataloader,
        val_batch_dataloader=val_batch_dataloader,
        num_epochs=config["mil_epochs"]
    )

    # Once trained, load parameters and train with those as the initialisation parameters
    parameter_state_dict = model.state_dict()
    true_model = AttentionNetworkCoord_32(-1, -1)
    true_model.load_state_dict(parameter_state_dict, strict=False)

    dataset = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        initial_pixel_cropper=TrainingPixelROI(480 // divisor_height, 640 // divisor_width, add_spatial_maps=True),
        transform=preprocessing_transforms,
        force_not_cache=True
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
    # tip_velocity_estimator.plot_train_val_losses()
