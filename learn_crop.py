import argparse
import pprint

from stable_baselines import PPO2, SAC
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import CallbackList
from stable_baselines.common.vec_env import DummyVecEnv

from lib.common.test_utils import get_distance_between_boxes
from lib.common.utils import set_up_cuda, get_preprocessing_transforms, get_seed
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.networks import AttentionNetworkCoord_32
from lib.rl.callbacks import CropScoreCallback
from lib.rl.demonstration_env import CropDemonstrationEnv, TestRewardSingleDemonstrationEnv
from lib.rl.demonstration_eval import CropEvaluator
from lib.rl.policies import PPOPolicy, SACCustomPolicy
from lib.rl.utils import DatasetModality

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--score_every", type=int, required=True)
    parser.add_argument("--images_every", type=int, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--tile", required=True)
    parser.add_argument("--env_type", required=True)
    parser.add_argument("--val_dem", required=True)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--target_updates", type=int, default=128)
    parser.add_argument("--training", type=float, default=0.8)
    parser.add_argument("--restrict_crop_move", type=int)
    parser.add_argument("--init_from", )
    parse_result = parser.parse_args()

    dataset = "scene1/scene1"
    config = dict(
        n_envs=16,
        buffer_size=parse_result.buffer_size,
        target_updates=parse_result.target_updates,
        size=(128, 96),
        cropped_size=(32, 24),
        learning_rate=0.0001,
        network_klass=AttentionNetworkCoord_32,
        seed=get_seed("random"),
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        split=[parse_result.training, 0.1, 0.1],
        patience=10,
        max_epochs=100,
        validate_epochs=1,
        name=parse_result.name,
        log_dir="learn_crop_output_log",
        add_coord=parse_result.version == "coord",
        tile=parse_result.tile == "yes",
        shuffle=True,
        val_dem=parse_result.val_dem,
        restrict_crop_move=parse_result.restrict_crop_move
    )

    device = set_up_cuda(config["seed"])
    config["device"] = device
    preprocessing_transforms, transforms = get_preprocessing_transforms(config["size"])

    print("Config:")
    pprint.pprint(config)

    dataset = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        transform=preprocessing_transforms,
        force_cache=True
    )

    evaluator = None
    if parse_result.env_type == "test":
        print("Test environment selected")
        env = TestRewardSingleDemonstrationEnv(
            demonstration_dataset=dataset,
            config=config
        )
    else:
        print("Estimator training environment selected")
        num_val_demonstrations = int(config["split"][1] * dataset.get_num_demonstrations())
        config["val_dem"] = num_val_demonstrations if config["val_dem"] == "all" else int(config["val_dem"])

        test_env = CropDemonstrationEnv(
            demonstration_dataset=dataset,
            config=config,
            dataset_type_idx=DatasetModality.VALIDATION,
            skip_reward=True
        )

        evaluator = CropEvaluator(
            test_env=test_env,
            num_iter=config["val_dem"]
        )

        env = CropDemonstrationEnv(
            demonstration_dataset=dataset,
            config=config,
            init_from=parse_result.init_from,
            evaluator=evaluator
        )

    monitor = Monitor(env=env, filename=f"{config['log_dir']}/")
    if parse_result.algo == "ppo":
        dummy = DummyVecEnv([lambda: monitor])
        model = PPO2(
            PPOPolicy,
            dummy,
            policy_kwargs=dict(image_size=config["size"], add_coord=config["add_coord"], tile=config["tile"]),
            verbose=1,
            gamma=1.0,
            tensorboard_log=f"./{config['log_dir']}"
        )
    elif parse_result.algo == "sac":
        model = SAC(
            SACCustomPolicy,
            monitor,
            policy_kwargs=dict(image_size=config["size"], add_coord=config["add_coord"], tile=config["tile"]),
            verbose=1,
            gamma=1.0,
            buffer_size=config["buffer_size"],
            target_update_interval=config["target_updates"],
            tensorboard_log="./learn_crop_output_log"
        )
    else:
        raise ValueError("Invalid algorithm, please choose ppo or sac")

    if evaluator is not None:
        evaluator.set_rl_model(rl_model=model)

    score_callback_train = CropScoreCallback(
        score_name="tl_distance_train",
        score_function=get_distance_between_boxes,
        prefix=f"{config['name']}_train",
        log_dir=f"{config['log_dir']}/train_{parse_result.algo}",
        config=config,
        demonstration_dataset=dataset,
        crop_test_modality=DatasetModality.TRAINING,
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
