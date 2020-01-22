import os
import shutil

import numpy as np
from pyrep.errors import ConfigurationPathError
from pyrep.objects.shape import Shape

from lib.camera_robot import CameraRobot
from lib.scenes import CameraBackgroundObjectsExtendedTextureReachCubeScene
from lib.utils import save_images_and_tip_velocities

if __name__ == "__main__":

    seed = 2019
    np.random.seed(seed)

    with CameraBackgroundObjectsExtendedTextureReachCubeScene(headless=True) as pr:

        # Minimum number of training samples we want to generate
        min_samples = 4000
        # count of number of training samples so far (image, tip velocity)
        total_count = 0
        # Number of the demonstration
        demonstration_num = 0
        folder = "./text_camera_background_v2"
        tip_velocity_file = "velocities.csv"
        metadata_file = "metadata.json"
        # remove data folder to regenerate data. Alternatively, change this to write to a different folder
        shutil.rmtree(folder, ignore_errors=True)
        os.mkdir(folder)
        os.chdir(folder)
        robot = CameraRobot(pr, show_paths=True)
        print("Position camera:", robot.movable_camera.get_position())
        target_cube = Shape("target_cube")
        target_cube_position = target_cube.get_position()
        print("Target cube position -0.05:", target_cube_position)
        target_above_cube = (np.array(target_cube_position) + np.array([0.0, 0.0, 0.05])).tolist()

        while total_count < min_samples:
            print("{}/{} samples generated".format(total_count, min_samples))
            offset = robot.generate_offset() if total_count > 0 else np.zeros(robot.generate_offset().shape[0])
            print("Offset {}".format(offset))
            try:
                tip_positions, tip_velocities, images, crop_pixels = robot.generate_image_simulation(
                    offset=offset, target=target_above_cube, target_object=target_cube
                )
                print(tip_positions[0])
                # save_images_and_tip_velocities(
                #     images=images,
                #     demonstration_num=demonstration_num,
                #     tip_positions=tip_positions,
                #     tip_velocities=tip_velocities,
                #     tip_velocity_file=tip_velocity_file,
                #     metadata_file=metadata_file,
                #     crop_pixels=crop_pixels
                # )
                demonstration_num += 1
                total_count += len(tip_velocities)
            except ConfigurationPathError:
                print("Error, can not reach object from offset: {}, ignoring...".format(offset))
                break
