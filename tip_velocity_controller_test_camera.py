import math

import numpy as np
from pyrep.objects.shape import Shape

from lib import utils
from lib.camera_robot import CameraRobot
from lib.controller import TipVelocityController, IdentityCropper, TruePixelROI, ControllerType
from lib.scenes import CameraBackgroundObjectsTextureReachCubeSceneV3
import json
import time

from lib.tip_velocity_estimator import TipVelocityEstimator

if __name__ == "__main__":

    models = [
        "AttentionNetworkCoordRand",
        "AttentionNetworkV3RandL",
        "AttentionNetworkV3RandL1",
        "AttentionNetworkV3RandL2",
        "AttentionNetworkV3RandL3",
        "AttentionNetworkV3RandL4",
    ]
    for model_name in models:
        with CameraBackgroundObjectsTextureReachCubeSceneV3(headless=True) as (pr, scene):
            camera_robot = CameraRobot(pr)
            test = "test_offsets_random.json"
            target_cube = Shape("target_cube")
            target_above_cube = np.array(target_cube.get_position()) + np.array([0.0, 0.0, 0.05])

            cropper = TruePixelROI(480 // 2, 640 // 2, camera_robot.get_movable_camera(), target_cube,
                                   add_spatial_maps=True)
            # cropper = IdentityCropper()
            c_type = ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            controller = TipVelocityController(
                tve_model=TipVelocityEstimator.load("models/{}.pt".format(model_name)),
                target_object=target_cube,
                camera=camera_robot.get_movable_camera(),
                roi_estimator=cropper,
                controller_type=c_type
            )
            print(controller.get_model().test_loss)

            test_name = "{}-{}.camera_robot.test".format(model_name, int(time.time()))
            result_json = {"min_distances": {}, "errors": {}}

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
                    scene=scene
                )
                count += 1
                # for index, i in enumerate(images):
                # utils.save_image(i, "/home/pablo/Desktop/t-{}image{}.png".format(count, index))
                result_json["min_distances"][str(idx)] = result["min_distance"]
                result_json["errors"][str(idx)] = dict(
                    combined_errors=result["combined_errors"],
                    velocity_errors=result["velocity_errors"],
                    orientation_errors=result["orientation_errors"]
                )

                if result["achieved"]:
                    achieved_count += 1

                print("Min distance: ", result["min_distance"])
                print("Achieved: ", achieved_count / (idx + 1), "{}/{}".format(achieved_count, idx + 1))

            print("Achieved: ", achieved_count / len(json_offset_list))

            with open(test_name, "w") as f:
                f.write(json.dumps(result_json))
