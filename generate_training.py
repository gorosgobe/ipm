import csv
import json
import os
import shutil

import cv2
import numpy as np
from pyrep.errors import ConfigurationPathError
from pyrep.objects.shape import Shape

from lib.camera_robot import CameraRobot
from lib.scenes import CameraTextureReachCubeScene


def save_images(images, format_str, prefix=""):
    for idx, img in enumerate(images):
        img = cv2.convertScaleAbs(img, alpha=(255.0))
        cv2.imwrite(format_str.format(prefix, idx), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def save_images_and_tip_velocities(images, demonstration_num, tip_positions, tip_velocities, tip_velocity_file, metadata_file, crop_pixels):
    format_str = "{}image{}.png"
    prefix = str(demonstration_num)
    save_images(images=images, format_str=format_str, prefix=prefix)

    with open(tip_velocity_file, "a", newline='') as f:
        writer = csv.writer(f)
        for idx, vel in enumerate(tip_velocities):
            writer.writerow([format_str.format(prefix, idx), *vel])

    if not os.path.exists(metadata_file):
        with open(metadata_file, "x"):
            pass

    with open(metadata_file, "r") as metadata_json:
        content = metadata_json.read()
        if not content:
            content = "{}"
        data = json.loads(content)

    with open(metadata_file, "w+") as metadata_json:
        if "demonstrations" not in data:
            data["demonstrations"] = {}

        start = data["demonstrations"][str(demonstration_num - 1)]["end"] + 1 if str(demonstration_num - 1) in data[
            "demonstrations"] else 0

        data["demonstrations"][demonstration_num] = dict(
            num_tip_velocities=len(tip_velocities),
            start=start,
            end=start + len(tip_velocities) - 1,
            tip_positions=tip_positions,
            crop_pixels=crop_pixels
        )

        if "num_demonstrations" not in data:
            data["num_demonstrations"] = 0
        data["num_demonstrations"] += 1
        metadata_json.write(json.dumps(data))


if __name__ == "__main__":

    seed = 2019
    np.random.seed(seed)

    with CameraTextureReachCubeScene(headless=True) as pr:

        # Minimum number of training samples we want to generate
        min_samples = 4000
        # count of number of training samples so far (image, tip velocity)
        total_count = 0
        # Number of the demonstration
        demonstration_num = 0
        folder = "./text_camera_unit"
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
                save_images_and_tip_velocities(
                    images=images,
                    demonstration_num=demonstration_num,
                    tip_positions=tip_positions,
                    tip_velocities=tip_velocities,
                    tip_velocity_file=tip_velocity_file,
                    metadata_file=metadata_file,
                    crop_pixels=crop_pixels
                )
                demonstration_num += 1
                total_count += len(tip_velocities)
            except ConfigurationPathError:
                print("Error, can not reach object from offset: {}, ignoring...".format(offset))
                break
