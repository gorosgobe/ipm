import argparse
import pprint

import numpy as np
import torch
from stable_baselines import PPO2, SAC
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import CallbackList
from stable_baselines.common.vec_env import DummyVecEnv

from lib.common.test_utils import get_distance_between_boxes
from lib.common.utils import set_up_cuda, get_preprocessing_transforms, get_seed
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.networks import AttentionNetworkCoord_32
from lib.rl.callbacks import ScoreCallback
from lib.rl.demonstration_env import SingleDemonstrationEnv, TestRewardSingleDemonstrationEnv
from lib.rl.policies import PPOPolicy, SACCustomPolicy
from lib.rl.utils import CropTestModality

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--score_every", type=int, required=True)
    parser.add_argument("--images_every", type=int, required=True)
    parser.add_argument("--epochs_reward", type=int, required=True)
    parser.add_argument("--epochs_validate", type=int, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--env_type", required=True)
    parser.add_argument("--init_from", )
    parse_result = parser.parse_args()

    dataset = "scene1/scene1"
    config = dict(
        n_envs=16,
        size=(128, 96),
        cropped_size=(32, 24),
        learning_rate=0.0001,
        network_klass=AttentionNetworkCoord_32,
        seed=get_seed("random"),
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        num_workers=2,  # number of workers to compute RL reward
        split=[0.8, 0.1, 0.1],
        patience=10,
        max_epochs=parse_result.epochs_reward,
        validate_epochs=parse_result.epochs_validate,
        name=parse_result.name,
        log_dir="learn_crop_output_log",
        add_coord=parse_result.version == "coord",
        shuffle=True
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

    if parse_result.env_type == "test":
        print("Test environment selected")
        env = TestRewardSingleDemonstrationEnv(
            demonstration_dataset=dataset,
            config=config
        )
    else:
        print("Estimator training environment selected")
        env = SingleDemonstrationEnv(
            demonstration_dataset=dataset,
            config=config,
            init_from=parse_result.init_from
        )

    monitor = Monitor(env=env, filename=f"{config['log_dir']}/")
    if parse_result.algo == "ppo":
        dummy = DummyVecEnv([lambda: monitor])
        model = PPO2(
            PPOPolicy,
            dummy,
            policy_kwargs=dict(image_size=config["size"], add_coord=config["add_coord"]),
            verbose=1,
            gamma=1.0,
            tensorboard_log=f"./{config['log_dir']}"
        )
    elif parse_result.algo == "sac":
        model = SAC(
            SACCustomPolicy,
            monitor,
            policy_kwargs=dict(image_size=config["size"], add_coord=config["add_coord"]),
            verbose=1,
            gamma=1.0,
            tensorboard_log="./learn_crop_output_log"
        )
    else:
        raise ValueError("Invalid algorithm, please choose ppo or sac")

    score_callback_train = ScoreCallback(
        score_name="tl_distance_train",
        score_function=get_distance_between_boxes,
        prefix=f"{config['name']}_train",
        log_dir=f"{config['log_dir']}/train_{parse_result.algo}",
        config=config,
        demonstration_dataset=dataset,
        crop_test_modality=CropTestModality.TRAINING,
        compute_score_every=parse_result.score_every,  # every rollout, for the time being
        number_rollouts=1,
        save_images_every=parse_result.images_every
    )
    model.learn(total_timesteps=parse_result.timesteps, callback=CallbackList([score_callback_train]))

    try:
        print("Finished training, mean #epochs trained:", env.get_epoch_list_stats())
    except (ValueError, NotImplementedError):
        print("Finished training, no estimators were trained")

    model.save(config["name"])

    try:
        # save validation losses obtained
        env.save_validation_losses_list(f"{config['name']}_val_losses")
    except NotImplementedError:
        print("Could not save validation losses, no estimators were trained")
