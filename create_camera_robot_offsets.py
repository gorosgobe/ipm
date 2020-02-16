import json

import numpy as np

from lib.camera_robot import CameraRobot
from lib.scenes import CameraBackgroundObjectsTextureReachCubeSceneV3, CameraScene5

if __name__ == '__main__':
    with CameraScene5(headless=True) as (pr, scene):
        camera_robot = CameraRobot(pr)
        target_position_above_cube = np.array(scene.get_target().get_position()) + np.array([0.0, 0.0, 0.05])
        np.random.seed(2019)
        content = {"offset": [], "distractor_positions": []}
        for i in range(100):
            offset = camera_robot.generate_offset()
            distractor_positions = camera_robot.set_distractor_random_positions(scene, target_position_above_cube)
            content["offset"].append(list(offset))
            content["distractor_positions"].append(distractor_positions)

        file_contents = json.dumps(content)

        with open("scene5_test.json", "w+") as f:
            f.write(file_contents)
