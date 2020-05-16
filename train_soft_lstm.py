import argparse
import os

from ax import optimize
from torch.utils.data import DataLoader

from lib.common.utils import get_preprocessing_transforms, set_up_cuda, get_demonstrations, get_seed
from lib.soft.soft_manager import SoftManager
from lib.soft.soft_rnn_dataset import SoftRNNDataset


def evaluation_function(parameterization, dataset, config, train_data_loader, validation_data_loader):
    entropy_lambda = parameterization["entropy_lambda"]
    manager = SoftManager(
        name=config["name"],
        dataset=dataset,
        hidden_size=config["hidden_size"],
        device=config["device"],
        is_coord=config["is_coord"],
        entropy_lambda=entropy_lambda
    )

    manager.train(
        num_epochs=config["max_epochs"],
        train_dataloader=train_data_loader,
        val_dataloader=validation_data_loader
    )
    return manager.get_best_val_loss()


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
    parser.add_argument("--entropy_lambda", type=float, default=1.0)
    parser.add_argument("--is_bop", default="no")
    parser.add_argument("--hidden_size", type=int, default=64)
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
        is_coord=parse_result.is_coord == "yes",
        entropy_lambda=parse_result.entropy_lambda,
        is_bop=parse_result.is_bop == "yes",
        hidden_size=parse_result.hidden_size
    )

    print("Name:", config["name"])
    device = set_up_cuda(config["seed"])
    config["device"] = device
    preprocessing_transforms, transforms = get_preprocessing_transforms(config["size"])

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

    if config["is_bop"]:
        best_parameters, values, experiment, model = optimize(
            parameters=[
                dict(
                    name="entropy_lambda", type="range", bounds=[1e-4, 1.0]
                )
            ],
            evaluation_function=lambda params: evaluation_function(params, dataset=dataset, config=config,
                                                                   train_data_loader=train_data_loader,
                                                                   validation_data_loader=validation_data_loader),
            total_trials=30,
            minimize=True
        )
        print("Best entropy parameter:")
        print(best_parameters)
        print("Means, covariances")
        means, covariances = values
        print(means)
        print(covariances)
    else:
        manager = SoftManager(
            name=config["name"],
            dataset=dataset,
            hidden_size=config["hidden_size"],
            device=config["device"],
            is_coord=config["is_coord"],
            entropy_lambda=config["entropy_lambda"]
        )

        manager.train(
            num_epochs=config["max_epochs"],
            train_dataloader=train_data_loader,
            val_dataloader=validation_data_loader
        )

        manager.plot_attention_on(train_data_loader, f"{config['name']}-train", 0)
        manager.plot_attention_on(train_data_loader, f"{config['name']}-train", 1)
        manager.plot_attention_on(validation_data_loader, f"{config['name']}-val", 0)
        manager.plot_attention_on(validation_data_loader, f"{config['name']}-val", 1)

        manager.save_best_model(os.path.join(config["save_to_location"], "soft_lstm"))
