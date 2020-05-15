import json
import os

import numpy as np
import torch
from stable_baselines import SAC, PPO2

from lib.common.test_utils import get_testing_configs, get_scene_and_test_scene_configuration, TestConfig
from lib.cv.controller import TipVelocityController
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.dsae.dsae import CustomDeepSpatialAutoencoder, DSAE_Encoder
from lib.dsae.dsae_action_predictor import ActionPredictorManager
from lib.dsae.dsae_action_tvec_adapter import DSAETipVelocityEstimatorAdapter
from lib.dsae.dsae_feature_provider import FeatureProvider, FilterSpatialRLFeatureProvider
from lib.dsae.dsae_manager import DSAEManager
from lib.dsae.dsae_networks import TargetVectorDSAE_Decoder
from lib.simulation.camera_robot import CameraRobot
from soft.rnn_tvec_adapter import RNNTipVelocityControllerAdapter


def get_full_image_networks(scene_trained, training_list, prefix=""):
    models_res = []
    for tr in training_list:
        models_res.append(f"{prefix}FullImageNetwork_{scene_trained}_{tr}")

    for tr in training_list:
        for sc in ["32", "64"]:
            for v in ["a", "coord"]:
                models_res.append(f"{prefix}FullImageNetwork_{scene_trained}_{v}_{sc}_{tr}")

    for tr in training_list:
        models_res.append(f"{prefix}FullImageNetwork_{scene_trained}_coord_{tr}")

    return models_res, TestConfig.FULL_IMAGE


def get_baseline_networks(scene_trained, training_list, prefix=""):
    models_res = []
    for tr in training_list:
        models_res.append(f"{prefix}BaselineNetwork_{scene_trained}_{tr}")
    return models_res, TestConfig.BASELINE


def get_attention_networks(size, scene_trained, training_list, prefix=""):
    models_res = []
    if size == 64:
        config = TestConfig.ATTENTION_64
    elif size == 32:
        config = TestConfig.ATTENTION_32
    else:
        raise ValueError("Unknown size")

    for tr in training_list:
        for v in ["V1", "V2", "tile"]:
            if size == 64:
                models_res.append(f"{prefix}AttentionNetwork{v}_{scene_trained}_{tr}")
            else:
                models_res.append(f"{prefix}AttentionNetwork{v}_{scene_trained}_32_{tr}")

    return models_res, config


def get_coord_attention_networks(size, scene_trained, training_list, prefix=""):
    models_res = []
    if size == 64:
        config = TestConfig.ATTENTION_COORD_64
    elif size == 32:
        config = TestConfig.ATTENTION_COORD_32
    else:
        raise ValueError("Unknown size")

    for tr in training_list:
        if size == 64:
            models_res.append(f"{prefix}AttentionNetworkcoord_{scene_trained}_{tr}")
        else:
            models_res.append(f"{prefix}AttentionNetworkcoord_{scene_trained}_32_{tr}")

    return models_res, config


def get_dsae_tve_model(dsae_path, action_predictor_path, latent_dimension, k, image_output_size=(24, 32), rl_path=None,
                       is_sac=None):
    if (rl_path is None) + (is_sac is None) == 1:
        raise ValueError("rl_path and is_sac should be both None or both contain information about the RL model")

    state_dict = DSAEManager.load_state_dict(os.path.join("models/dsae", dsae_path))
    model = CustomDeepSpatialAutoencoder(
        encoder=DSAE_Encoder(
            in_channels=3,
            out_channels=(latent_dimension * 2, latent_dimension, latent_dimension // 2),
            strides=(2, 1, 1),
            normalise=True
        ),
        decoder=TargetVectorDSAE_Decoder(
            image_output_size=image_output_size,
            latent_dimension=latent_dimension,
            normalise=True
        )
    )
    model.load_state_dict(state_dict)
    device = torch.device("cpu")

    if rl_path is None:
        feature_provider = FeatureProvider(model=model, device=device)
    else:
        path = os.path.join("models/rl", rl_path)
        if is_sac:
            rl_model = SAC.load(path)
        else:
            rl_model = PPO2.load(path)

        feature_provider = FilterSpatialRLFeatureProvider(
            feature_provider_model=model, device=device, rl_model=rl_model, k=k
        )

    action_predictor = ActionPredictorManager.load(
        path=os.path.join("models/dsae/action_predictor", action_predictor_path),
        k=k
    )

    dsae_tve_model = DSAETipVelocityEstimatorAdapter(feature_provider=feature_provider,
                                                     action_predictor=action_predictor)
    return dsae_tve_model


def get_default_tve_model(tve_prefix, tve_model_name):
    return TipVelocityEstimator.load("models/{}{}.pt".format(tve_prefix, tve_model_name))


def get_recurrent_tve_model(tve_prefix, tve_model_name):
    return RNNTipVelocityControllerAdapter.load(os.path.join("models", tve_prefix, tve_model_name))


if __name__ == "__main__":

    trainings = [
        "005",
        "010",
        "015",
        "02",
        "04",
        "08"
    ]

    scenes = ["scene1scene1"]

    vs = ["V4"]

    sizes = ["64", "32"]

    # models, testing_config_name = get_full_image_networks(scenes[0], trainings)
    # models, testing_config_name = get_baseline_networks(scenes[0], trainings)
    # models, testing_config_name = get_attention_networks(32, scenes[0], trainings)
    # models, testing_config_name = get_coord_attention_networks(32, scenes[0], trainings)
    # for scene in scenes:
    #     for t in trainings:
    #         # for ty in types:
    #         #     for size in sizes:
    #         models.append(f"FullImageNetwork_{scene}_{t}")
    # models = []
    # for v in vs:
    #     models.extend([
    #         f"FullImageNetworkWD{v}_scene1scene1_08",
    #         f"FullImageNetworkWD{v}_scene1scene1_04",
    #         f"FullImageNetworkWD{v}_scene1scene1_02",
    #         f"FullImageNetworkWD{v}_scene1scene1_015",
    #         f"FullImageNetworkWD{v}_scene1scene1_010",
    #         f"FullImageNetworkWD{v}_scene1scene1_005",
    #     ])
    #
    # testing_config_name = TestConfig.ATTENTION_COORD_ROT_32
    # models = [
    #     "AttentionNetworkcoordRot_scene1scene1_32_08",
    #     "AttentionNetworkcoordRot_scene1scene1_32_04",
    #     "AttentionNetworkcoordRot_scene1scene1_32_02",
    #     "AttentionNetworkcoordRot_scene1scene1_32_015",
    #     "AttentionNetworkcoordRot_scene1scene1_32_010",
    #     "AttentionNetworkcoordRot_scene1scene1_32_005",
    #
    # ]
    # testing_config_name = TestConfig.ATTENTION_COORD_32
    # prefix = "fixed_steps_datasets/"
    # models = ["FullImageNetworkSoft_scene1scene1_08_v1", "FullImageNetworkSoft_scene1scene1_08_v2", "FullImageNetworkSoft_scene1scene1_08_v3"]

    testing_config_name = TestConfig.DSAE
    # # define if network is dsae type
    dsae_path = "target_64_0001_1_1_08_scene1scene1_v1.pt"
    models = ["act_ppo_dense_k3_64_0001_08_scene1scene1.pt"]
    latent_dimension = 128
    k = 3
    # # for RL based versions
    # rl_path = "ppo_test_v2"
    rl_path = "ppo_dense_k3"
    is_sac = False
    # is_sac = False

    for model_name in models:
        s, test = get_scene_and_test_scene_configuration(model_name=model_name)
        with s(headless=True) as (pr, scene):
            camera_robot = CameraRobot(pr)
            target_cube = scene.get_target()
            target_above_cube = np.array(target_cube.get_position()) + np.array([0.0, 0.0, 0.05])

            testing_configs = get_testing_configs(camera_robot=camera_robot, target_cube=target_cube)
            testing_config = testing_configs[testing_config_name]

            cropper = testing_config["cropper"]
            c_type = testing_config["c_type"]
            if testing_config_name == TestConfig.DSAE:
                print("DSAE-based policy!")
                tve_model = get_dsae_tve_model(
                    dsae_path=dsae_path,
                    action_predictor_path=model_name,
                    latent_dimension=latent_dimension,
                    k=k,
                    rl_path=rl_path,
                    is_sac=is_sac
                )
            elif testing_config_name == TestConfig.RECURRENT_FULL:
                print("Recurrent policy!")
                tve_model = get_recurrent_tve_model(prefix, model_name)
            else:
                print("Default type policy!")
                tve_model = get_default_tve_model(prefix, model_name)
            controller = TipVelocityController(
                tve_model=tve_model,
                target_object=target_cube,
                camera=camera_robot.get_movable_camera(),
                roi_estimator=cropper,
                controller_type=c_type
            )
            try:
                m = controller.get_model().network

                print("Parameters:", sum([p.numel() for p in m.parameters()]))
                print("Model name:", model_name)
                print("Network: ", m)
                print(controller.get_model().test_loss)
            except Exception:
                print("Parameters not directly available for this network...")

            test_name = "{}.test".format(model_name)
            result_json = {"min_distances": {}, "errors": {}, "fixed_steps_distances": {}}

            with open(test, "r") as f:
                content = f.read()
                js_content = json.loads(content)
                json_offset_list = js_content["offset"]
                distractor_positions_list = js_content["distractor_positions"]

            achieved_count = 0
            count = 0
            for idx, offset in enumerate(json_offset_list):
                print("Offset:", offset)
                distractor_positions = distractor_positions_list[idx]
                print("Distractor positions:", distractor_positions)
                controller.start()
                result = camera_robot.run_controller_simulation(
                    controller=controller,
                    offset=np.array(offset),
                    target=target_above_cube,
                    distractor_positions=distractor_positions,
                    scene=scene,
                    fixed_steps=scene.get_steps_per_demonstration()
                )
                count += 1
                # for index, i in enumerate(result["images"]):
                #     save_image(i, "/home/pablo/Desktop/rl-{}image{}.png".format(count, index))
                result_json["min_distances"][str(idx)] = result["min_distance"]
                result_json["fixed_steps_distances"][str(idx)] = result["fixed_steps_distance"]
                result_json["errors"][str(idx)] = dict(
                    combined_errors=result["combined_errors"],
                    velocity_errors=result["velocity_errors"],
                    orientation_errors=result["orientation_errors"]
                )

                if result["achieved"]:
                    achieved_count += 1

                print("Min distance: ", result["min_distance"])
                print("Fixed step distance: ", result["fixed_steps_distance"])
                print("Achieved: ", achieved_count / (idx + 1), "{}/{}".format(achieved_count, idx + 1))

            print("Achieved: ", achieved_count / len(json_offset_list))

            with open(test_name, "w") as f:
                f.write(json.dumps(result_json))
