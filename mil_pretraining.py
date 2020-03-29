from collections import OrderedDict

import torch
from torchmeta.utils.data import BatchMetaDataLoader, MetaDataLoader, SubsetTask

from dataset import ImageTipVelocitiesDataset
from meta_dataset import MILTipVelocityDataset, DatasetType
from meta_networks import MetaNetwork, MetaAttentionNetworkCoord
from mil import MetaImitationLearning, MetaAlgorithm
from utils import set_up_cuda, get_seed, get_preprocessing_transforms


if __name__ == '__main__':

    config = dict(
        size=(32, 24),
        seed="random",
        location="models/pretraining_test",
        meta_algorithm=MetaAlgorithm.FOMAML,
        adaptation_steps=1,
        step_size=0.1,
        optimizer=torch.optim.Adam,
        learning_rate=0.01,
        loss_function=torch.nn.MSELoss(),
        max_batches=10,
        split=[0.8, 0.1, 0.1]
    )

    device = set_up_cuda(seed=get_seed(parsed_seed=config["seed"]))
    preprocessing_transforms, transforms = get_preprocessing_transforms(size=config["size"])

    model = MetaAttentionNetworkCoord()
    mil = MetaImitationLearning(
        model=model,
        meta_algorithm=config["meta_algorithms"],
        num_adaptation_steps=config["num_adaptation_steps"],
        step_size=config["step_size"],
        optimizer=config["optimizer"](model.parameters(), lr=config["learning_rate"]),
        loss_function=config["loss_function"],
        max_batches=config["max_batches"],
        device=device
    )

    dataset = "scene1/scene1"
    dataset = ImageTipVelocitiesDataset(
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        transform=preprocessing_transforms
    )

    meta_train_dataset = MILTipVelocityDataset(
        demonstration_dataset=dataset,
        split=config["split"],
        dataset_type=DatasetType.TRAIN
    )

    meta_val_dataset = MILTipVelocityDataset(
        demonstration_dataset=dataset,
        split=config["split"],
        dataset_type=DatasetType.VAL
    )

    train_batch_dataloader = BatchMetaDataLoader(meta_train_dataset, batch_size=5,
                                            num_workers=8)
    val_batch_dataloader = BatchMetaDataLoader(meta_val_dataset, batch_size=6, num_workers=8)

    mil.train(
        train_batch_dataloader=train_batch_dataloader,
        val_batch_dataloader=val_batch_dataloader,
        num_epochs=1
    )
