import argparse
import os

import torch
from stable_baselines import SAC, PPO2
from torch.utils.data import DataLoader
from torchvision import transforms
from lib.common.utils import get_seed, set_up_cuda, get_demonstrations
from lib.dsae.dsae import CustomDeepSpatialAutoencoder, DSAE_Encoder
from lib.dsae.dsae_action_predictor import ActionPredictorManager
from lib.dsae.dsae_dataset import DSAE_FeatureProviderDataset
from lib.dsae.dsae_feature_provider import FeatureProvider, FilterSpatialRLFeatureProvider
from lib.dsae.dsae_manager import DSAEManager
from lib.dsae.dsae_networks import DSAE_TargetActionPredictor, TargetVectorDSAE_Decoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dsae_path", required=True)
    parser.add_argument("--latent", type=int, required=True)
    parser.add_argument("--training", type=float, required=True)

    # if rl path is not provided, then we train action predictor with features from DSAE only
    # otherwise, we use the top k features as selected by the RL agent
    parser.add_argument("--rl_path")
    parser.add_argument("--rl_algo")
    parser.add_argument("--k", type=int)
    parser.add_argument("--seed", default="random")
    # set to 2 or 4
    parser.add_argument("--output_divisor", type=int, default=4)
    parse_result = parser.parse_args()

    if (parse_result.rl_path is None) + (parse_result.k is None) + (parse_result.rl_algo is None) not in [0, 3]:
        raise ValueError("If RL provide rl model path, k and RL algorithm. DSAE with all features is used otherwise.")

    if parse_result.rl_algo is not None and parse_result.rl_algo not in ["sac", "ppo"]:
        raise ValueError("Only SAC and PPO RL algorithms are supported")

    seed = get_seed(parse_result.seed)
    device = set_up_cuda(seed)
    dataset_name = parse_result.dataset

    config = dict(
        lr=0.001,
        device=device,
        name=parse_result.name,
        seed=seed,
        epochs=parse_result.epochs,
        dataset_name=dataset_name,
        dsae_path=parse_result.dsae_path,
        latent_dimension=parse_result.latent,
        output_divisor=parse_result.output_divisor,
        split=[0.8, 0.1, 0.1],
        training=0.8,
        size=(96, 128),
        batch_size=32,
        k=parse_result.k,
        rl_path=parse_result.rl_path,
        is_sac=parse_result.rl_algo == "sac"
    )

    height, width = config["size"]
    reduce_grayscale = transforms.Compose([
        transforms.Resize(size=(height // config["output_divisor"], width // config["output_divisor"])),
        transforms.Grayscale()
    ])

    model = CustomDeepSpatialAutoencoder(
        encoder=DSAE_Encoder(
            in_channels=3,
            out_channels=(config["latent_dimension"] * 2, config["latent_dimension"], config["latent_dimension"] // 2),
            strides=(2, 1, 1),
            normalise=True
        ),
        decoder=TargetVectorDSAE_Decoder(
            image_output_size=(height // config["output_divisor"], width // config["output_divisor"]),
            latent_dimension=config["latent_dimension"],
            normalise=True
        )
    )
    model.load_state_dict(DSAEManager.load_state_dict(os.path.join("models/dsae", config["dsae_path"])))

    if config["rl_path"] is None:
        # DSAE case with all features
        # CUDA gives errors if using GPU with Dataloaders -> for faster training, we cache the data
        feature_provider = FeatureProvider(model=model, device=None)
    else:
        path = os.path.join("models/rl", config["rl_path"])
        if config["is_sac"]:
            rl_model = SAC.load(path)
        else:
            rl_model = PPO2.load(path)

        feature_provider = FilterSpatialRLFeatureProvider(
            feature_provider_model=model, device=None, rl_model=rl_model, k=config["k"]
        )

    dataset = DSAE_FeatureProviderDataset(
        feature_provider=feature_provider,
        root_dir=dataset_name,
        velocities_csv=f"{dataset_name}/velocities.csv",
        metadata=f"{dataset_name}/metadata.json",
        rotations_csv=f"{dataset_name}/rotations.csv",
        input_resize_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(height, width))
        ]),
        reduced_transform=reduce_grayscale,
        size=config["size"],
        cache=True
    )

    training_demonstrations, validation_demonstrations, _test_demonstrations \
        = get_demonstrations(dataset, config["split"], limit_train_coeff=config["training"])

    dataloader = DataLoader(dataset=training_demonstrations, batch_size=config["batch_size"], shuffle=True,
                            num_workers=4)
    validation_dataloader = DataLoader(dataset=validation_demonstrations, batch_size=config["batch_size"], shuffle=True,
                                       num_workers=4)

    action_predictor = DSAE_TargetActionPredictor(
        # k can be set to all features normally, or to the k parameter if used with an RL agent
        k=(config["latent_dimension"] // 2) if config["k"] is None else config["k"]
    )
    optimiser = torch.optim.Adam(action_predictor.parameters(), lr=config["lr"])
    manager = ActionPredictorManager(
        action_predictor=action_predictor,
        num_epochs=config["epochs"],
        optimiser=optimiser,
        device=config["device"],
        verbose=True,
        name=config["name"]
    )
    manager.train(train_dataloader=dataloader, validation_dataloader=validation_dataloader)
    manager.save_best_model("models/dsae/action_predictor/")
