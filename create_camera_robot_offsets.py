import json

import numpy as np

from lib.camera_robot import CameraRobot

if __name__ == '__main__':
    np.random.seed(2019)
    content = {"offset": []}
    for i in range(100):
        offset = CameraRobot.generate_offset()
        content["offset"].append(list(offset))

    file_contents = json.dumps(content)

    with open("test_offsets_orientations.json", "w+") as f:
        f.write(file_contents)
