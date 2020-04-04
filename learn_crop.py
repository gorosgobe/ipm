import pprint

import numpy as np
import torch
from stable_baselines import PPO2, results_plotter, SAC
from stable_baselines.bench import Monitor
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader

from lib.common.utils import set_up_cuda, get_preprocessing_transforms, get_seed, get_demonstrations
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.rl.demonstration_env import SingleDemonstrationEnv
from lib.cv.networks import FullImageNetwork_32

if __name__ == '__main__':
    dataset = "scene1/scene1"
    config = dict(
        n_envs=16,
        size=(128, 96),
        cropped_size=(32, 24),
        learning_rate=0.0001,
        network_klass=FullImageNetwork_32,
        seed=get_seed("random"),
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        num_workers=4,  # number of workers to compute RL reward
        split=[0.2, 0.1, 0.1],
        patience=3,  # smaller, need to train faster
        max_epochs=50,
        validate_epochs=1
    )
    print("Config:")
    pprint.pprint(config)

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = set_up_cuda(config["seed"])
    config["device"] = device
    preprocessing_transforms, transforms = get_preprocessing_transforms(config["size"])

    dataset = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        transform=preprocessing_transforms,
    )

    env = SingleDemonstrationEnv(
        demonstration_dataset=dataset,
        config=config
    )

    env = Monitor(env=env, filename="learn_crop_output_log/")
    env = DummyVecEnv([lambda: env])

    model = PPO2(MlpPolicy, env, verbose=1, gamma=1.0, tensorboard_log="./learn_crop_output_log")

    # model = SAC(MlpPolicy, env, n_steps=32, verbose=1, gamma=1.0, tensorboard_log="./learn_crop_output_log")
    model.learn(total_timesteps=40960)
    # results_plotter.plot_results(["./learn_crop_output_log"], 1e4, results_plotter.X_TIMESTEPS, "Output")
