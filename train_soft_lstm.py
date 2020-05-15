import argparse
import os

from torch.utils.data import DataLoader

from lib.common.utils import get_preprocessing_transforms, set_up_cuda, get_demonstrations, get_seed
from lib.soft.soft_manager import SoftManager
from lib.soft.soft_rnn_dataset import SoftRNNDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", default="random")
    parser.add_argument("--training", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--is_coord", default="no")
    parse_result = parser.parse_args()

    seed = get_seed(parse_result.seed)
    print("Seed:", seed)
    dataset = parse_result.dataset
    print("Dataset: ", dataset)

    config = dict(
        seed=seed,
        size=(128, 96),
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        batch_size=parse_result.batch_size,
        split=[0.8, parse_result.val, 0.2 - parse_result.val],
        name=parse_result.name,
        max_epochs=parse_result.epochs,
        validate_epochs=1,
        save_to_location="models/",
        is_coord=parse_result.is_coord == "yes"
    )

    print("Name:", config["name"])
    device = set_up_cuda(config["seed"])
    config["device"] = device
    # TODO: CoordConv not used for the time being, consider as an extra
    preprocessing_transforms, transforms = get_preprocessing_transforms(config["size"], is_coord=False)

    dataset = SoftRNNDataset(
        velocities_csv=config["velocities_csv"],
        metadata=config["metadata"],
        rotations_csv=config["rotations_csv"],
        root_dir=config["root_dir"],
        cache=True,
        transform=preprocessing_transforms,
        is_coord=config["is_coord"]
    )

    limit_training_coefficient = parse_result.training or 0.8  # all training data
    print("Training coeff limit:", limit_training_coefficient)

    training_demonstrations, val_demonstrations, _ = get_demonstrations(
        dataset, config["split"], limit_training_coefficient
    )

    train_data_loader = DataLoader(training_demonstrations, batch_size=config["batch_size"], num_workers=4,
                                   shuffle=True, collate_fn=SoftRNNDataset.custom_collate)
    validation_data_loader = DataLoader(val_demonstrations, batch_size=32, num_workers=4, shuffle=True,
                                        collate_fn=SoftRNNDataset.custom_collate)

    manager = SoftManager(
        name=config["name"],
        dataset=dataset,
        hidden_size=64,
        device=config["device"],
        is_coord=config["is_coord"]
    )

    manager.train(
        num_epochs=config["max_epochs"],
        train_dataloader=train_data_loader,
        val_dataloader=validation_data_loader
    )

    manager.save_best_model(os.path.join(config["save_to_location"], "soft_lstm"))
