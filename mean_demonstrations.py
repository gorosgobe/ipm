import json
import os

import cv2
import numpy as np
import pandas as pd

if __name__ == '__main__':
    dataset = "text_camera_orient"
    os.chdir(dataset)
    frame = pd.read_csv("velocities.csv", header=None, usecols=[0])

    with open("metadata.json", "r") as f:
        metadata_content = f.read()

    metadata = json.loads(metadata_content)
    n_demonstrations = metadata["num_demonstrations"]
    for number in range(25):
        image_mean = np.zeros((480, 640, 3))
        count = 0
        for i in range(n_demonstrations):
            end = metadata["demonstrations"][str(i)]["num_tip_velocities"] - 1
            name = "{}image{}.png".format(i, number)
            image_path = os.path.join(os.getcwd(), name)
            # load image
            try:
                image = cv2.imread(image_path)
            except:
                continue
            if image is None:
                continue
            image_mean += image
            count += 1

        print("Completed", number)
        image_mean /= count
        cv2.imwrite(os.path.join("/home/pablo/Desktop", dataset + "_" + str(number) + ".png"), image_mean)
