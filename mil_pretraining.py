from collections import OrderedDict

import torch
from torchmeta.utils.data import BatchMetaDataLoader, MetaDataLoader, SubsetTask

from dataset import ImageTipVelocitiesDataset
from meta_dataset import MILTipVelocityDataset, DatasetType
from meta_networks import MetaNetwork, MetaAttentionNetworkCoord
from mil import MetaImitationLearning, MetaAlgorithm
from utils import set_up_cuda, get_seed, get_preprocessing_transforms


if __name__ == '__main__':
    location = "models/pretraining_test"
    seed = "random"  # TODO: change accordingly
    device = set_up_cuda(seed=get_seed(parsed_seed=seed))
    size = (32, 24)
    preprocessing_transforms, transforms = get_preprocessing_transforms(size=size)
    model = MetaAttentionNetworkCoord()
    mil = MetaImitationLearning(
        model=model,
        meta_algorithm=MetaAlgorithm.FOMAML,
        num_adaptation_steps=1,
        step_size=0.1,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        loss_function=torch.nn.MSELoss(),
        max_batches=10,
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

    split = [0.8, 0.1, 0.1]

    meta_train_dataset = MILTipVelocityDataset(
        demonstration_dataset=dataset,
        split=split,
        dataset_type=DatasetType.TRAIN
    )

    meta_val_dataset = MILTipVelocityDataset(
        demonstration_dataset=dataset,
        split=split,
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
