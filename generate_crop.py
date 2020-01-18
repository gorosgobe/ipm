import cv2
import pandas as pd
import os
import shutil

from lib.controller import OffsetCropper

if __name__ == '__main__':
    parent_dir = os.getcwd()
    base_dir = "./text_datalimitedshift30"
    folder = "./text_improvedcropdatalimitedshift30"
    shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)
    shutil.copyfile(os.path.join(base_dir, "velocities.csv"), os.path.join(folder, "velocities.csv"))
    shutil.copyfile(os.path.join(base_dir, "metadata.json"), os.path.join(folder, "metadata.json"))
    total_pixels_height = 480
    total_pixels_width = 640
    cropped_size_height = total_pixels_height // 2
    cropped_size_width = total_pixels_width // 2
    cropper = OffsetCropper(cropped_height=cropped_size_height, cropped_width=cropped_size_width, offset_height=25)
    os.chdir(base_dir)
    frame = pd.read_csv("velocities.csv", header=None, usecols=[0])
    for i in range(len(frame)):
        image_name = frame.iloc[i, 0]
        image_path = os.path.join(os.getcwd(), image_name)
        # load image
        image = cv2.imread(image_path)
        # crop image
        cropped_image = cropper.crop(image)
        # save in folder
        cv2.imwrite(os.path.join(parent_dir, folder, image_name), cropped_image)

