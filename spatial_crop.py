import argparse
import os
import pprint

from stable_baselines import PPO2, SAC
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import CallbackList
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.sac import MlpPolicy
from torchvision import transforms

from lib.common.test_utils import get_distance_between_boxes
from lib.common.utils import get_seed, set_up_cuda
from lib.dsae.dsae import CustomDeepSpatialAutoencoder, DSAE_Encoder
from lib.dsae.dsae_dataset import DSAE_FeatureProviderDataset
from lib.dsae.dsae_feature_provider import FeatureProvider
from lib.dsae.dsae_manager import DSAEManager
from lib.dsae.dsae_networks import TargetVectorDSAE_Decoder
from lib.networks import AttentionNetworkCoord_32
from lib.rl.callbacks import CropScoreCallback
from lib.rl.demonstration_env import SpatialFeatureCropEnv
from lib.rl.demonstration_eval import CropEvaluator, RandomCropEvaluator
from lib.rl.utils import DatasetModality

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--score_every", type=int, required=True)
    parser.add_argument("--images_every", type=int, required=True)
    parser.add_argument("--latent", type=int, required=True)
    parser.add_argument("--val_dem", required=True)
    parser.add_argument("--dsae_path", required=True)
    parser.add_argument("--random_val_crop", required=True)
    parser.add_argument("--training", type=float, default=0.8)
    parser.add_argument("--restrict_crop_move", type=int)
    parser.add_argument("--output_divisor", type=int, default=4)
    parse_result = parser.parse_args()

    dataset = "scene1/scene1"
    config = dict(
        size=(128, 96),
        cropped_size=(32, 24),
        network_klass=AttentionNetworkCoord_32,
        seed=get_seed("random"),
        latent_dimension=parse_result.latent,
        output_divisor=parse_result.output_divisor,
        dsae_path=parse_result.dsae_path,
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        split=[parse_result.training, 0.1, 0.1],
        name=parse_result.name,
        log_dir="spatial_crop_output_log",
        shuffle=True,
        val_dem=parse_result.val_dem,
        random_val_crop=parse_result.random_val_crop == "yes",
        restrict_crop_move=parse_result.restrict_crop_move
    )

    device = set_up_cuda(config["seed"])
    config["device"] = device
    print("Config:")
    pprint.pprint(config)

    width, height = config["size"]

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
    model.load_state_dict(DSAEManager.load_state_dict(os.path.join("models/dsae/", config["dsae_path"])))
    model.to(config["device"])
    feature_provider = FeatureProvider(model=model, device=config["device"])

    reduce_grayscale = transforms.Compose([
        transforms.Resize(size=(height // config["output_divisor"], width // config["output_divisor"])),
        transforms.Grayscale()
    ])

    dataset = DSAE_FeatureProviderDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        size=(height, width),
        feature_provider=feature_provider,
        reduced_transform=reduce_grayscale,
        input_resize_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(height, width))
        ]),
        cache=True,
        add_pixel=True,
        add_image=True
    )

    num_val_demonstrations = int(config["split"][1] * dataset.get_num_demonstrations())
    config["val_dem"] = num_val_demonstrations if config["val_dem"] == "all" else int(config["val_dem"])

    test_env = SpatialFeatureCropEnv(
        demonstration_dataset=dataset,
        latent_dimension=config["latent_dimension"],
        cropped_size=config["cropped_size"],
        split=config["split"],
        device=config["device"],
        network_klass=config["network_klass"],
        dataset_type_idx=DatasetModality.VALIDATION,
        skip_reward=True,
        restrict_crop_move=config["restrict_crop_move"]
    )

    if config["random_val_crop"]:
        evaluator = RandomCropEvaluator(
            test_env=test_env,
            num_iter=config["val_dem"]
        )
    else:
        evaluator = CropEvaluator(
            test_env=test_env,
            num_iter=config["val_dem"]
        )

    env = SpatialFeatureCropEnv(
        demonstration_dataset=dataset,
        latent_dimension=config["latent_dimension"],
        cropped_size=config["cropped_size"],
        split=config["split"],
        device=config["device"],
        network_klass=config["network_klass"],
        dataset_type_idx=DatasetModality.TRAINING,
        evaluator=evaluator,
        restrict_crop_move=config["restrict_crop_move"]
    )

    monitor = Monitor(env=env, filename=f"{config['log_dir']}/")
    if parse_result.algo == "ppo":
        dummy = DummyVecEnv([lambda: monitor])
        model = PPO2(
            "MlpPolicy",
            dummy,
            verbose=1,
            gamma=1.0,
            tensorboard_log=config["log_dir"]
        )
    elif parse_result.algo == "sac":
        model = SAC(
            MlpPolicy,
            monitor,
            verbose=1,
            gamma=1.0,
            buffer_size=1000000,
            tensorboard_log=config["log_dir"]
        )
    else:
        raise ValueError("Invalid algorithm, please choose ppo or sac")

    if evaluator is not None:
        evaluator.set_rl_model(rl_model=model)

    # separate test environment for callback
    score_test_env = SpatialFeatureCropEnv(
        demonstration_dataset=dataset,
        latent_dimension=config["latent_dimension"],
        cropped_size=config["cropped_size"],
        split=config["split"],
        device=config["device"],
        network_klass=config["network_klass"],
        dataset_type_idx=DatasetModality.VALIDATION,
        skip_reward=True,
        restrict_crop_move=config["restrict_crop_move"]
    )

    score_callback_train = CropScoreCallback(
        score_name="tl_distance_train",
        score_function=get_distance_between_boxes,
        prefix=f"{config['name']}_train",
        log_dir=f"{config['log_dir']}/train_{parse_result.algo}",
        config=config,
        compute_score_every=parse_result.score_every,
        number_rollouts=1,
        save_images_every=parse_result.images_every,
        test_env=score_test_env
    )
    model.learn(total_timesteps=parse_result.timesteps, callback=CallbackList([score_callback_train]))
    model.save(os.path.join("models/rl", config["name"]))
