import numpy as np
from pyrep.objects.shape import Shape

from lib import utils
from lib.camera_robot import CameraRobot
from lib.controller import TipVelocityController, IdentityCropper, TruePixelROI, ControllerType
from lib.scenes import CameraTextureReachCubeScene, CameraBackgroundObjectsTextureReachCubeScene, \
    CameraBackgroundObjectsExtendedTextureReachCubeScene
import json
import time

from lib.tip_velocity_estimator import TipVelocityEstimator

if __name__ == "__main__":

    with CameraBackgroundObjectsTextureReachCubeScene(headless=True) as pr:
        camera_robot = CameraRobot(pr)
        model_name = "BaselineNetworkL3"
        test = "test_offsets_orientations.json"
        target_cube = Shape("target_cube")
        target_above_cube = np.array(target_cube.get_position()) + np.array([0.0, 0.0, 0.05])

        cropper = TruePixelROI(480//2, 640//2, camera_robot.get_movable_camera(), target_cube)
        #cropper = IdentityCropper()
        c_type = ControllerType.RELATIVE_POSITION_AND_ORIENTATION
        controller = TipVelocityController(
            tve_model=TipVelocityEstimator.load("models/{}.pt".format(model_name)),
            target_object=target_cube,
            camera=camera_robot.get_movable_camera(),
            roi_estimator=cropper,
            controller_type=c_type
        )
        print(controller.get_model().test_loss)

        test_name = "{}-{}.camera_robot.test".format(model_name, int(time.time()))
        min_distances = {}

        with open(test, "r") as f:
            content = f.read()
            json_offset_list = json.loads(content)["offset"]

        achieved_count = 0
        count = 0
        for idx, offset in enumerate(json_offset_list):
            print("Offset:", offset)
            images, tip_velocities, achieved, min_distance = camera_robot.run_controller_simulation(
                controller=controller, offset=np.array(offset), target=target_above_cube)
            count += 1
            #for index, i in enumerate(images):
                #utils.save_image(i, "/home/pablo/Desktop/t-{}image{}.png".format(count, index))
            min_distances[str(idx)] = min_distance

            if achieved:
                achieved_count += 1
            
            print("Min distance: ", min_distance)
            print("Achieved: ", achieved_count / (idx + 1), "{}/{}".format(achieved_count, idx + 1))

        print("Achieved: ", achieved_count / len(json_offset_list))

        with open(test_name, "w") as f:
            f.write(json.dumps(min_distances))
