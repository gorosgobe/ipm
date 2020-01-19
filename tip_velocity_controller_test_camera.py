import numpy as np
from pyrep.objects.shape import Shape

from lib.camera_robot import CameraRobot
from lib.controller import TipVelocityController, IdentityCropper, TruePixelROI
from lib.scenes import CameraTextureReachCubeScene
import json
import time

from lib.tip_velocity_estimator import TipVelocityEstimator

if __name__ == "__main__":

    with CameraTextureReachCubeScene(headless=True) as pr:
        camera_robot = CameraRobot(pr)
        model_name = "M22_unit"
        target_cube = Shape("target_cube")
        target_above_cube = np.array(target_cube.get_position()) + np.array([0.0, 0.0, 0.5])

        #cropper = TruePixelROI(480//2, 640//2, camera_robot.movable_camera, target_cube)
        cropper = IdentityCropper()
        controller = TipVelocityController(TipVelocityEstimator.load("models/{}.pt".format(model_name)), cropper)
        print(controller.get_model().test_loss)

        test_name = "{}-{}.camera_robot.test".format(model_name, int(time.time()))
        min_distances = {}

        with open("test_offsets.json", "r") as f:
            content = f.read()
            json_offset_list = json.loads(content)["offset"]

        achieved_count = 0
        for idx, offset in enumerate(json_offset_list):
            print(offset)
            try:
                images, tip_velocities, achieved, min_distance = camera_robot.run_controller_simulation(
                    controller=controller, offset=np.array(offset), target=target_above_cube)
                min_distances[str(idx)] = min_distance
            except Exception as exc:
                print("Exception ocurred: ", exc)
                achieved = False

            if achieved:
                achieved_count += 1
            print("Min distance: ", min_distance)
            print("Achieved: ", achieved_count / (idx + 1), "{}/{}".format(achieved_count, idx + 1))

        print("Achieved: ", achieved_count / len(json_offset_list))

        with open(test_name, "w") as f:
            f.write(json.dumps(min_distances))
