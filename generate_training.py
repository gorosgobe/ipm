import os
import shutil

import numpy as np

from lib.simulation.camera_robot import CameraRobot
from lib.simulation.scenes import CameraScene1, CameraScene5, CameraScene4, CameraScene3
from lib.common.utils import save_images_and_tip_velocities

if __name__ == "__main__":

    seed = 4045
    np.random.seed(seed)

    with CameraScene5(headless=True, random_distractors=True) as (pr, scene):

        # Minimum number of training samples we want to generate
        # 5250
        min_samples = 5250
        # usually True, unless testing with simpler datasets
        discontinuity = True
        # count of number of training samples so far (image, tip velocity)
        total_count = 0
        counts = []
        # Number of the demonstration
        demonstration_num = 0
        folder = "./randist5"
        tip_velocity_file = "velocities.csv"
        rotations_file = "rotations.csv"
        metadata_file = "metadata.json"
        # remove data folder to regenerate data. Alternatively, change this to write to a different folder
        shutil.rmtree(folder, ignore_errors=True)
        os.mkdir(folder)
        os.chdir(folder)
        robot = CameraRobot(pr, show_paths=False)
        print("Position camera:", robot.movable_camera.get_position())
        target_cube = scene.get_target()
        target_cube_position = target_cube.get_position()
        print("Target cube position -0.05:", target_cube_position)
        target_above_cube = (np.array(target_cube_position) + np.array([0.0, 0.0, 0.05])).tolist()
        stop_demonstration = []
        while total_count < min_samples:
            print("{}/{} samples generated".format(total_count, min_samples))
            offset = robot.generate_offset() if total_count > 0 else np.zeros(robot.generate_offset().shape[0])
            print("Offset {}".format(offset))
            result = robot.generate_image_simulation(
                scene=scene, offset=offset, target_position=target_above_cube, target_object=target_cube,
                randomise_distractors=True, discontinuity=discontinuity
            )
            save_images_and_tip_velocities(
                demonstration_num=demonstration_num,
                tip_velocity_file=tip_velocity_file,
                metadata_file=metadata_file,
                rotations_file=rotations_file,
                **result
            )
            demonstration_num += 1
            total_count += len(result["tip_velocities"])
            counts.append(len(result["tip_velocities"]))
            stop_demonstration.append(
                result["count_stop_demonstration"] if result["count_stop_demonstration"] is not None else -1)

        print("Counts mean:", np.array(counts).mean())
        print("Stop demonstration mean:", np.array(stop_demonstration).mean())
