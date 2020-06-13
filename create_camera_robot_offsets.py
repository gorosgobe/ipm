import json

import numpy as np

from distractor import Distractor
from sawyer_camera_adapter import DiscOffsetGenerator, SawyerCameraAdapter
from simulation.camera_robot import CameraRobot
from simulation.scenes import CameraScene4, SawyerInsertDiscScene

if __name__ == '__main__':
    with SawyerInsertDiscScene(headless=True, random_distractors=True) as (pr, scene):
        camera_robot = CameraRobot(pr, movable=SawyerCameraAdapter(pr), offset_generator=DiscOffsetGenerator())
        # make sure random distractors is set to True if you plan to use random distractors later when testing
        num_distractors = len(scene.get_distractors())
        # target_distractor_position = np.array(scene.get_target().get_position()) + np.array([0.0, 0.0, 0.05])
        target_distractor_position = scene.get_target().get_target_for_distractor_point().get_position()
        # content = {"offset": [], "distractor_positions": []}
        content = {"offset": [], "test_distractors": []}
        for _ in range(100):
            offset = camera_robot.generate_offset()
            distractors = []
            for _ in range(num_distractors):
                random_distractor = Distractor()
                rand_dsp = random_distractor.get_safe_distance()
                distractors.append((random_distractor, rand_dsp))
            # distractors = scene.get_distractors()
            # dsp = scene.get_distractor_safe_distances()
            # set safe distances
            _distractor_positions = camera_robot.set_distractor_random_positions(target_distractor_position, *zip(*distractors))
            distractors = list(map(lambda dpr: dpr[0].serialise(), distractors))

            content["offset"].append(list(offset))

            # content["distractor_positions"].append(distractor_positions)
            content["test_distractors"].append(distractors)

        file_contents = json.dumps(content)

        with open("test_demonstrations/disc_test.json", "w+") as f:
            f.write(file_contents)
