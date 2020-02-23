import math

import numpy as np

from lib import utils
from lib.camera_robot import CameraRobot
from lib.controller import TipVelocityController, IdentityCropper, TruePixelROI, ControllerType
from lib.scenes import CameraScene1, CameraScene2, CameraScene3, CameraScene4, CameraScene5
import json
from lib.tip_velocity_estimator import TipVelocityEstimator

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

    models = []
    for scene in scenes:
        for t in trainings:
            # for ty in types:
            #     for size in sizes:
            models.append(f"AttentionNetworkcoord_{scene}_{t}")

    models = ["test_full_discontinuity_scene1"]
    testing_config_name = "fullimage"

    prefix = "" #"fixed_steps_datasets/"

    for model_name in models:
        if "scene2" in model_name:
            s = CameraScene2
            test = "scene2_test.json"
        elif "scene3" in model_name:
            s = CameraScene3
            test = "scene3_test.json"
        elif "scene4" in model_name:
            s = CameraScene4
            test = "scene4_test.json"
        elif "scene5" in model_name:
            s = CameraScene5
            test = "scene5_test.json"
        elif "rand" in model_name.lower() or "scene1" in model_name:
            s = CameraScene1
            test = "test_offsets_random.json"

        with s(headless=True) as (pr, scene):
            camera_robot = CameraRobot(pr)
            target_cube = scene.get_target()
            target_above_cube = np.array(target_cube.get_position()) + np.array([0.0, 0.0, 0.05])

            testing_configs = {
                "baseline":
                    {
                        "cropper": IdentityCropper(),
                        "c_type": ControllerType.RELATIVE_POSITION_AND_ORIENTATION
                    },
                "fullimage":
                    {
                        "cropper": IdentityCropper(),
                        "c_type": ControllerType.DEFAULT
                    },
                "attention_64":
                    {
                        "cropper": TruePixelROI(480 // 2, 640 // 2, camera_robot.get_movable_camera(), target_cube),
                        "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
                    },
                "attention_32":
                    {
                        "cropper": TruePixelROI(480 // 4, 640 // 4, camera_robot.get_movable_camera(), target_cube),
                        "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
                    },
                "attention_coord_64":
                    {
                        "cropper": TruePixelROI(480 // 2, 640 // 2, camera_robot.get_movable_camera(), target_cube,
                                                add_spatial_maps=True),
                        "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
                    },
                "attention_coord_32":
                    {
                        "cropper": TruePixelROI(480 // 4, 640 // 4, camera_robot.get_movable_camera(), target_cube,
                                                add_spatial_maps=True),
                        "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
                    },
            }

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
                #     utils.save_image(i, "/home/pablo/Desktop/f-{}image{}.png".format(count, index))
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
