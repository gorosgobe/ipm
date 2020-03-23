import json

import numpy as np

from lib.camera_robot import CameraRobot
from lib.controller import TipVelocityController
from lib.tip_velocity_estimator import TipVelocityEstimator
from lib.utils import get_testing_configs, get_scene_and_test_scene_configuration, TestConfig


def get_full_image_networks(scene_trained, training_list):
    models_res = []
    for tr in training_list:
        models_res.append(f"FullImageNetwork_{scene_trained}_{tr}")

    for tr in training_list:
        for sc in ["32", "64"]:
            for v in ["a", "coord"]:
                models_res.append(f"FullImageNetwork_{scene_trained}_{v}_{sc}_{tr}")

    for tr in training_list:
        models_res.append(f"FullImageNetwork_{scene_trained}_coord_{tr}")

    return models_res, TestConfig.FULL_IMAGE


def get_baseline_networks(scene_trained, training_list):
    models_res = []
    for tr in training_list:
        models_res.append(f"BaselineNetwork_{scene_trained}_{tr}")
    return models_res, TestConfig.BASELINE


def get_attention_networks(size, scene_trained, training_list):
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
                models_res.append(f"AttentionNetwork{v}_{scene_trained}_{tr}")
            else:
                models_res.append(f"AttentionNetwork{v}_{scene_trained}_32_{tr}")

    return models_res, config


def get_coord_attention_networks(size, scene_trained, training_list):
    models_res = []
    if size == 64:
        config = TestConfig.ATTENTION_COORD_64
    elif size == 32:
        config = TestConfig.ATTENTION_COORD_32
    else:
        raise ValueError("Unknown size")

    for tr in training_list:
        if size == 64:
            models_res.append(f"AttentionNetworkcoord_{scene_trained}_{tr}")
        else:
            models_res.append(f"AttentionNetworkcoord_{scene_trained}_32_{tr}")

    return models_res, config


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

    vs = ["V1", "V2"]

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
    models = [
        "AttentionNetworkcoord_scene1scene1_compV1_32_04",
        "AttentionNetworkcoord_scene1scene1_compV1_32_02",
        "AttentionNetworkcoord_scene1scene1_compV1_32_015",
        "AttentionNetworkcoord_scene1scene1_compV1_32_010",
        "AttentionNetworkcoord_scene1scene1_compV1_32_005"
    ]
    testing_config_name = TestConfig.ATTENTION_COORD_32

    prefix = "composite_loss_test/"

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
            controller = TipVelocityController(
                tve_model=TipVelocityEstimator.load("models/{}{}.pt".format(prefix, model_name)),
                target_object=target_cube,
                camera=camera_robot.get_movable_camera(),
                roi_estimator=cropper,
                controller_type=c_type
            )
            m = controller.get_model().network

            print("Parameters:", sum([p.numel() for p in m.parameters()]))
            print("Model name:", model_name)
            print("Network: ", m)
            print(controller.get_model().test_loss)

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
                #     utils.save_image(i, "/home/pablo/Desktop/baseline-{}image{}.png".format(count, index))
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
