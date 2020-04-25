import argparse
import os
import pprint

from stable_baselines import PPO2, SAC
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import CallbackList
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac import MlpPolicy
from torchvision import transforms

from lib.rl.callbacks import FeatureDistanceScoreCallback
from lib.common.utils import get_seed, set_up_cuda
from lib.dsae. dsae import CustomDeepSpatialAutoencoder, DSAE_Encoder
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
    parser.add_argument("--algo", required=True)
    parser.add_argument("--dsae_path", required=True)
    parser.add_argument("--latent", type=int, required=True)
    parser.add_argument("--score_every", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--val_dem", required=True)

    parser.add_argument("--seed", default="random")
    parser.add_argument("--n_steps", type=int, default=128)
    parser.add_argument("--num_avg_training", type=int, default=1)
    parser.add_argument("--training", type=float, default=0.8)
    # set to 2 or 4
    parser.add_argument("--output_divisor", type=int, default=4)
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
        k=parse_result.k,
        algo=parse_result.algo,
        split=[parse_result.training, 0.2, 0.0],
        n_steps=parse_result.n_steps,
        score_every=parse_result.score_every,
        dsae_path=parse_result.dsae_path,
        num_avg_training=parse_result.num_avg_training,
        # number of demonstrations used as validation dataset during reward comp.
        val_dem=parse_result.val_dem,
        log_dir="filter_spatial_output_log/"
    )

    pprint.pprint(config)

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

    num_val_demonstrations = int(config["split"][1] * dataset.get_num_demonstrations())
    config["val_dem"] = num_val_demonstrations if config["val_dem"] == "all" else int(config["val_dem"])

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
    model.state_dict(DSAEManager.load_state_dict(os.path.join("models/dsae/", config["dsae_path"])))
    model.to(config["device"])
    feature_provider = FeatureProvider(model=model, device=config["device"])

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

    # 5 validation demonstrations for validation loss estimation
    evaluator = FilterSpatialEvaluator(test_env=validation_env, num_iter=config["val_dem"])
    env = FilterSpatialFeatureEnv(
        latent_dimension=config["latent_dimension"],
        feature_provider=feature_provider,
        demonstration_dataset=dataset,
        split=config["split"],
        dataset_type_idx=DatasetModality.TRAINING,
        device=config["device"],
        k=config["k"],
        num_average_training=config["num_avg_training"],
        evaluator=evaluator
    )
    monitor = Monitor(env=env, filename=config["log_dir"])
    if config["algo"] == "ppo":
        dummy = DummyVecEnv([lambda: monitor])
        rl_model = PPO2(
            "MlpPolicy",
            dummy,
            verbose=True,
            gamma=1.0,
            n_steps=config["n_steps"],
            nminibatches=4,
            noptepochs=4,
            tensorboard_log=config["log_dir"]
        )
    elif config["algo"] == "sac":
        rl_model = SAC(
            MlpPolicy,
            monitor,
            verbose=True,
            gamma=1.0,
            tensorboard_log=config["log_dir"]
        )
    else:
        raise ValueError("Unknown RL algorithm, only PPO and SAC are supported.")

    evaluator.set_rl_model(rl_model)
    rl_model.learn(config["timesteps"], callback=CallbackList([
        FeatureDistanceScoreCallback(
            test_env=validation_env,
            # to estimate on entire validation set
            n_episodes=num_val_demonstrations,
            every=config["score_every"]
        )
    ]))
    rl_model.save(config["name"])
