import cv2
import numpy as np
from pyrep import PyRep
from pyrep.backend import vrep
from pyrep.errors import ConfigurationPathError
from os.path import join, dirname, abspath

from sawyer_robot import SawyerRobot

SCENE_FILE = join(dirname(abspath(__file__)), "scenes/sawyer_reach_cube.ttt")

def save_images(images, prefix=""):
    for idx, img in enumerate(images):
        img = cv2.convertScaleAbs(img, alpha=(255.0))
        cv2.imwrite("{}image{}.png".format(prefix, idx), img)

pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()

sawyer_robot = SawyerRobot(pr)
tip_positions, images = sawyer_robot.run_simulation()

save_images(images)
for i in range(5):
    offset_angles_list = np.random.uniform(-np.pi / 10, np.pi / 10, size=7)
    offset_angles      = {idx + 1: angle_offset for idx, angle_offset in enumerate(offset_angles_list)}
    print("Offset angles {}".format(offset_angles_list))
    try:
        sawyer_robot.run_simulation(offset_angles=offset_angles)
    except ConfigurationPathError:
        print("Error, can not reach object from offset: {}, ignoring...".format(offset_angles))

pr.stop()
pr.shutdown()
