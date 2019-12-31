import json
import os

import cv2
import numpy as np
import pandas as pd

if __name__ == '__main__':
    dataset = "text_improvedcropdatalimitedshift30"
    os.chdir(dataset)
    frame = pd.read_csv("velocities.csv", header=None, usecols=[0])

    with open("metadata.json", "r") as f:
        metadata_content = f.read()

    metadata = json.loads(metadata_content)
    n_demonstrations = metadata["num_demonstrations"]
    print(n_demonstrations)
    for number in range(25):
        image_mean = np.zeros((240, 320, 3))
        count = 0
        for i in range(n_demonstrations):
            end = metadata["demonstrations"][str(i)]["num_tip_velocities"] - 1
            print(end)
            name = "{}image{}.png".format(i, number)
            image_path = os.path.join(os.getcwd(), name)
            # load image
            try:
                image = cv2.imread(image_path)
            except:
                continue
            image_mean = image_mean + image
            count += 1

        print("Completed")
        image_mean /= count
        print(image_mean)
        cv2.imwrite(os.path.join("/home/pablo/Desktop", dataset + "_" + name + ".png"), image_mean)
