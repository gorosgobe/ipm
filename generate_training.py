import cv2
import numpy as np
from pyrep import PyRep
from pyrep.backend import vrep
from pyrep.robots.arms.sawyer import Sawyer
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
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
sawyer_robot.run_simulation()
sawyer_robot.run_simulation()

pr.stop()
pr.shutdown()
