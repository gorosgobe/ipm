import argparse
import pprint

import numpy as np
import torch
from stable_baselines import PPO2, SAC, sac
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import CallbackList
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import DummyVecEnv

from lib.common.utils import set_up_cuda, get_preprocessing_transforms, get_seed
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.cv.networks import AttentionNetworkCoord
from lib.rl.callbacks import ScoreCallback
from lib.rl.demonstration_env import SingleDemonstrationEnv
from lib.rl.policies import PPOPolicy
from lib.common.test_utils import get_distance_between_boxes
from lib.rl.utils import CropTestModality

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--name")
    parser.add_argument("--score_every", type=int)
    parse_result = parser.parse_args()

    dataset = "scene1/scene1"
    config = dict(
        n_envs=16,
        size=(128, 96),
        cropped_size=(64, 48),
        learning_rate=0.0001,
        network_klass=AttentionNetworkCoord,
        seed=get_seed("random"),
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        num_workers=4,  # number of workers to compute RL reward
        split=[0.2, 0.1, 0.1],
        patience=3,  # smaller, need to train faster
        max_epochs=100,
        validate_epochs=1,
        name=parse_result.name or "rl_crop",
        log_dir="learn_crop_output_log"
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

    print("Checking environment...")
    check_env(env)
    print("Check successful!")

    env = Monitor(env=env, filename=f"{config['log_dir']}/")
    if parse_result.algo == "ppo":
        dummy = DummyVecEnv([lambda: env])
        model = PPO2(PPOPolicy, dummy, policy_kwargs=dict(image_size=config["size"]), verbose=1, gamma=1.0,
                     tensorboard_log=f"./{config['log_dir']}")
    elif parse_result.algo == "sac":
        model = SAC(sac.MlpPolicy, env, verbose=1, gamma=1.0, tensorboard_log="./learn_crop_output_log")
    else:
        raise ValueError("Invalid algorithm, please choose ppo or sac")

    score_callback_train = ScoreCallback(
        score_name="tl_distance_train",
        score_function=get_distance_between_boxes,
        prefix="train",
        log_dir=f"{config['log_dir']}/train_{parse_result.algo}",
        config=config,
        demonstration_dataset=dataset,
        crop_test_modality=CropTestModality.TRAINING.value,
        compute_score_every=parse_result.score_every,  # every rollout, for the time being
        number_rollouts=1,
        save_images_every=10
    )
    model.learn(total_timesteps=parse_result.timesteps, callback=CallbackList([score_callback_train]))
    print("Finished training, mean #epochs trained:", np.mean(np.array(env.get_epoch_list())))
    model.save(config["name"])

# results_plotter.plot_results(["./learn_crop_output_log"], 1e4, results_plotter.X_TIMESTEPS, "Output")
