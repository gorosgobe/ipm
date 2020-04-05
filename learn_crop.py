import argparse
import pprint

import numpy as np
import torch
from stable_baselines import PPO2, SAC, sac
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from lib.cv.controller import TrainingPixelROI
from lib.common.utils import set_up_cuda, get_preprocessing_transforms, get_seed, ResizeTransform, save_image
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.cv.networks import AttentionNetworkCoord_32, AttentionNetworkCoord
from lib.rl.demonstration_env import SingleDemonstrationEnv
from lib.common.test_utils import draw_crop, calculate_IoU, downsample_coordinates, get_distance_between_boxes
from lib.cv.utils import CvUtils

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo")
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--name")
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
        name=parse_result.name or "rl_crop"
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
    if parse_result.algo == "ppo":
        dummy = DummyVecEnv([lambda: env])
        model = PPO2(MlpPolicy, dummy, verbose=1, gamma=1.0, tensorboard_log="./learn_crop_output_log")
    elif parse_result.algo == "sac":
        model = SAC(sac.MlpPolicy, env, verbose=1, gamma=1.0, tensorboard_log="./learn_crop_output_log")
    else:
        raise ValueError("Invalid algorithm, please choose ppo or sac")

    model.learn(total_timesteps=parse_result.timesteps)
    print("Finished training, mean #epochs trained:", np.mean(np.array(env.get_epoch_list())))
    model.save(config["name"])

    # WIP
    dataset_plain_images = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"]
    )

    resize = ResizeTransform((128, 96))

    dataset_get_crop_box = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        transform=resize,
        initial_pixel_cropper=TrainingPixelROI(
            480 // 2, 640 // 2, add_spatial_maps=False
        ),
        force_not_cache=True
    )

    obs = env.reset()
    done = False
    count = 0
    while not done:
        demonstration_index = env.demonstration_img_idx
        numpy_image_plain_gt = dataset_plain_images[demonstration_index]
        numpy_pixel_info = dataset_get_crop_box[demonstration_index]["pixel_info"]
        image_gt = numpy_image_plain_gt["image"].copy()
        tl_gt = numpy_pixel_info["top_left"].astype(int)
        br_gt = numpy_pixel_info["bottom_right"].astype(int)
        draw_crop(image_gt, tl_gt, br_gt, size=4)
        image_gt = resize(image_gt)

        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        center = info["center_crop_pixel"]
        cropped_width, cropped_height = config["cropped_size"]
        coords = CvUtils.get_bounding_box_coordinates(center[0], center[1], cropped_height=cropped_height,
                                                      cropped_width=cropped_width)
        box = CvUtils.get_bounding_box(*coords)
        predicted_pixel_info_tl = box[0]
        predicted_pixel_info_br = box[3]
        draw_crop(image_gt, predicted_pixel_info_tl, predicted_pixel_info_br, red=True)
        save_image(image_gt, f"imagetest-{count}.png")

        width, height = config["size"]
        tl_gt_down = downsample_coordinates(*tl_gt, og_width=640, og_height=480, to_width=width, to_height=height)
        score = get_distance_between_boxes(tl_gt_down, predicted_pixel_info_tl)
        print(f"Score for image {count} is {score}")
        count += 1

# results_plotter.plot_results(["./learn_crop_output_log"], 1e4, results_plotter.X_TIMESTEPS, "Output")
