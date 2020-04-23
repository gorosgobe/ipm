import argparse
import os

from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from torchvision import transforms

from lib.common.utils import get_seed, set_up_cuda
from lib.dsae.dsae import CustomDeepSpatialAutoencoder, DSAE_Encoder
from lib.dsae.dsae_dataset import DSAE_Dataset
from lib.dsae.dsae_feature_provider import FeatureProvider
from lib.dsae.dsae_manager import DSAEManager
from lib.dsae.dsae_networks import TargetVectorDSAE_Decoder
from lib.rl.demonstration_env import FilterSpatialFeatureEnv, FilterSpatialEvaluator
from lib.rl.utils import DatasetModality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", default="random")
    parser.add_argument("--dsae_path", required=True)
    parser.add_argument("--latent", type=int, required=True)
    # set to 2 or 4
    parser.add_argument("--output_divisor", type=int, required=True)
    parse_result = parser.parse_args()

    seed = get_seed(parse_result.seed)
    device = set_up_cuda(seed)
    dataset_name = parse_result.dataset

    config = dict(
        seed=seed,
        name=parse_result.name,
        device=device,
        size=(96, 128),
        latent_dimension=parse_result.latent,  # 64 features
        output_divisor=parse_result.output_divisor,
        timesteps=parse_result.timesteps,
        k=10,
        split=[0.8, 0.2, 0.0],
        dsae_path=parse_result.dsae_path,
        log_dir="filter_spatial_output_log/"
    )

    height, width = config["size"]
    # transform for comparison between real and outputted image
    reduce_grayscale = transforms.Compose([
        transforms.Resize(size=(height // config["output_divisor"], width // config["output_divisor"])),
        transforms.Grayscale()
    ])

    dataset = DSAE_Dataset(
        root_dir=dataset_name,
        velocities_csv=f"{dataset_name}/velocities.csv",
        metadata=f"{dataset_name}/metadata.json",
        rotations_csv=f"{dataset_name}/rotations.csv",
        input_resize_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(height, width))
        ]),
        reduced_transform=reduce_grayscale,
        size=config["size"]
    )

    model = CustomDeepSpatialAutoencoder(
        encoder=DSAE_Encoder(in_channels=3, out_channels=(64, 32, 16), strides=(2, 1, 1), normalise=True),
        decoder=TargetVectorDSAE_Decoder(
            image_output_size=(height // config["output_divisor"], width // config["output_divisor"]),
            latent_dimension=config["latent_dimension"],
            normalise=True
        )
    )
    model.state_dict(DSAEManager.load_state_dict(os.path.join("models/dsae/", config["dsae_path"])))
    model.to(config["device"])
    feature_provider = FeatureProvider(model=model)

    validation_env = FilterSpatialFeatureEnv(
        latent_dimension=config["latent_dimension"],
        feature_provider=feature_provider,
        demonstration_dataset=dataset,
        split=config["split"],
        dataset_type_idx=DatasetModality.VALIDATION,
        device=config["device"],
        k=config["k"],
        skip_reward=True
    )

    evaluator = FilterSpatialEvaluator(test_env=validation_env)
    env = FilterSpatialFeatureEnv(
        latent_dimension=config["latent_dimension"],
        feature_provider=feature_provider,
        demonstration_dataset=dataset,
        split=config["split"],
        dataset_type_idx=DatasetModality.TRAINING,
        device=config["device"],
        k=config["k"],
        evaluator=evaluator
    )
    monitor = Monitor(env=env, filename=config["log_dir"])
    dummy = DummyVecEnv([lambda: monitor])
    rl_model = PPO2(
        "MlpPolicy",
        dummy,
        verbose=True,
        gamma=1.0,
        n_steps=256,
        nminibatches=8,
        noptepochs=8,
        tensorboard_log=config["log_dir"]
    )

    evaluator.set_rl_model(rl_model)

    rl_model.learn(config["timesteps"])
    rl_model.save(config["name"])
