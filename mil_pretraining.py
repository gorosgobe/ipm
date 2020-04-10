import argparse
import pprint

import numpy as np
import torch
from torchmeta.utils.data import BatchMetaDataLoader

from lib.common.utils import set_up_cuda, get_seed, get_preprocessing_transforms, get_divisors, get_meta_algorithm
from lib.cv.controller import RandomPixelROI
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.meta.meta_dataset import MILTipVelocityDataset, DatasetType
from lib.meta.meta_networks import MetaAttentionNetworkCoord, MetaAttentionNetworkCoord_32
from lib.meta.mil import MetaImitationLearning

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--algo", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", default="random")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mil_epochs", type=int, required=True)
    parser.add_argument("--num_adaptation_steps", type=int, required=True)
    parser.add_argument("--num_tasks_in_batch", type=int, required=True)
    parser.add_argument("--step_size", type=float, required=True)
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
    algo = get_meta_algorithm(parse_result.algo)
    config = dict(
        name=parse_result.name,
        size=(128, 96),
        cropped_size=(32, 24),
        seed=seed,
        save_to_location="models/pretraining_test",
        mil_epochs=parse_result.mil_epochs,
        mil_meta_algorithm=algo,
        mil_adaptation_steps=parse_result.num_adaptation_steps,
        mil_num_tasks_in_batch=parse_result.num_tasks_in_batch,  # number of batches in inner loop
        mil_step_size=parse_result.step_size,
        mil_learning_rate=0.01,
        mil_max_batches=100,  # if multiplied by batch size of meta data loader -> number of tasks per epoch
        optimiser=torch.optim.Adam,
        loss_function=torch.nn.MSELoss(),
        split=[0.8, 0.1, 0.1],
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
    preprocessing_transforms, _ = get_preprocessing_transforms(size=config["cropped_size"], is_coord=True)

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

    divisor_width, divisor_height = get_divisors(*config["size"], *config["cropped_size"])
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

    train_batch_dataloader = BatchMetaDataLoader(meta_train_dataset, batch_size=config["mil_num_tasks_in_batch"], num_workers=8)
    val_batch_dataloader = BatchMetaDataLoader(meta_val_dataset, batch_size=config["mil_num_tasks_in_batch"], num_workers=8)

    mil.train(
        train_batch_dataloader=train_batch_dataloader,
        val_batch_dataloader=val_batch_dataloader,
        num_epochs=config["mil_epochs"]
    )

    mil.save_best_model(f"mil_{config['name']}.pt")