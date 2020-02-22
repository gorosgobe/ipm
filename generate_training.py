import os
import shutil

import numpy as np
from pyrep.errors import ConfigurationPathError
from pyrep.objects.shape import Shape

from lib.camera_robot import CameraRobot
from lib.scenes import CameraBackgroundObjectsTextureReachCubeSceneV3, CameraScene5
from lib.utils import save_images_and_tip_velocities

if __name__ == "__main__":

    seed = 2019
    np.random.seed(seed)

    with CameraScene5(headless=True) as (pr, scene):

        # Minimum number of training samples we want to generate
        min_samples = 4000
        # count of number of training samples so far (image, tip velocity)
        total_count = 0
        # Number of the demonstration
        demonstration_num = 0
        folder = "./scene5"
        tip_velocity_file = "velocities.csv"
        rotations_file = "rotations.csv"
        metadata_file = "metadata.json"
        # Every demonstration must be this long (in terms of observations)
        constant_steps_per_demonstration = 27
        # remove data folder to regenerate data. Alternatively, change this to write to a different folder
        shutil.rmtree(folder, ignore_errors=True)
        os.mkdir(folder)
        os.chdir(folder)
        robot = CameraRobot(pr, show_paths=True)
        print("Position camera:", robot.movable_camera.get_position())
        target_cube = scene.get_target()
        target_cube_position = target_cube.get_position()
        print("Target cube position -0.05:", target_cube_position)
        target_above_cube = (np.array(target_cube_position) + np.array([0.0, 0.0, 0.05])).tolist()

        while total_count < min_samples:
            print("{}/{} samples generated".format(total_count, min_samples))
            offset = robot.generate_offset() if total_count > 0 else np.zeros(robot.generate_offset().shape[0])
            print("Offset {}".format(offset))
            try:
                result = robot.generate_image_simulation(
                    scene=scene, offset=offset, target_position=target_above_cube, target_object=target_cube, randomise_distractors=True
                )
                save_images_and_tip_velocities(
                    demonstration_num=demonstration_num,
                    tip_velocity_file=tip_velocity_file,
                    metadata_file=metadata_file,
                    rotations_file=rotations_file,
                    augment_to_steps=constant_steps_per_demonstration,
                    **result
                )
                demonstration_num += 1
                total_count += len(result["tip_velocities"])
            except ConfigurationPathError:
                print("Error, can not reach object from offset: {}, ignoring...".format(offset))
                break
