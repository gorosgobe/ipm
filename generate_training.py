

import csv
import os
import shutil

import cv2
import numpy as np
from pyrep.errors import ConfigurationPathError

from sawyer_robot import SawyerRobot
from scenes import SawyerReachCubeScene


def save_images(images, format_str, prefix=""):
    for idx, img in enumerate(images):
        img = cv2.convertScaleAbs(img, alpha=(255.0))
        cv2.imwrite(format_str.format(prefix, idx), img)

def save_images_and_tip_velocities(images, demonstration_num, tip_velocities, tip_velocity_file):
    format_str = "{}image{}.png"
    prefix     = str(demonstration_num)
    save_images(images=images, format_str=format_str, prefix=prefix)

    with open(tip_velocity_file, "a", newline='') as f:
        writer = csv.writer(f)
        for idx, vel in enumerate(tip_velocities):
            writer.writerow([ format_str.format(prefix, idx), *vel])

if __name__ == "__main__":

    with SawyerReachCubeScene(headless=True) as pr:

        # Minimum number of training samples we want to generate
        min_samples = 10
        # count of number of training samples so far (image, tip velocity)
        total_count = 0
        # Number of the demonstration
        demonstration_num = 0
        folder = "./data123"
        # remove data folder to regenerate data. Alternatively, change this to write to a different folder
        shutil.rmtree(folder, ignore_errors=True)
        os.mkdir(folder)
        os.chdir(folder)
        sawyer_robot = SawyerRobot(pr)
        tip_positions, tip_velocities, images = sawyer_robot.run_simulation()
        print(tip_positions)
        print(tip_velocities)
        save_images_and_tip_velocities(
            images=images, 
            demonstration_num=demonstration_num, 
            tip_velocities=tip_velocities, 
            tip_velocity_file="velocities.csv"
        )
        total_count += len(tip_velocities)

        while total_count <= min_samples:
            print("{}/{} samples generated".format(total_count, min_samples))
            demonstration_num += 1
            offset_angles_list = np.random.uniform(-np.pi / 20, np.pi / 20, size=7)
            offset_angles      = {idx + 1: angle_offset for idx, angle_offset in enumerate(offset_angles_list)}
            print("Offset angles {}".format(offset_angles_list))
            try:
                tip_positions, tip_velocities, images = sawyer_robot.run_simulation(offset_angles=offset_angles)
                save_images_and_tip_velocities(
                    images=images, 
                    demonstration_num=demonstration_num, 
                    tip_velocities=tip_velocities, 
                    tip_velocity_file="velocities.csv"
                )
                total_count += len(tip_velocities)
            except ConfigurationPathError:
                print("Error, can not reach object from offset: {}, ignoring...".format(offset_angles))
                break
