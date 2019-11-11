import cv2
import numpy as np
from pyrep import PyRep
from pyrep.backend import vrep
from pyrep.robots.arms.sawyer import Sawyer
from os.path import join, dirname, abspath

SCENE_FILE = join(dirname(abspath(__file__)), "scenes/sawyer_reach_cube.ttt")

def get_vision_sensor():
    return vrep.simGetObjectHandle("sawyer_vision_sensor")

def save_images(images, prefix=""):
    for idx, img in enumerate(images):
        img = cv2.convertScaleAbs(img, alpha=(255.0))
        cv2.imwrite("{}image{}.png".format(prefix, idx), img)

def simulate_path(pr, path, sawyer):
    path.visualize()
    path.set_to_start()
    pr.step()

    vision_sensor = get_vision_sensor()
    resolution = vrep.simGetVisionSensorResolution(vision_sensor)

    tip_positions = []
    images = []
    done = False
    while not done:
        done = path.step()
        pr.step()
        tip_positions.append(sawyer.get_tip().get_position())
        images.append(vrep.simGetVisionSensorImage(vision_sensor, resolution))

    return tip_positions, images

pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()

sawyer = Sawyer()
tip = sawyer.get_tip()
tip_position = tip.get_position()
print(tip_position)

target_cube = vrep.simGetObjectHandle("target_cube")
target_cube_position = vrep.simGetObjectPosition(target_cube, -1)
print(target_cube_position)

sawyer_vision_sensor = vrep.simGetObjectHandle("sawyer_vision_sensor")
sawyer_vision_sensor_position = vrep.simGetObjectPosition(sawyer_vision_sensor, -1)
print(sawyer_vision_sensor_position)

# Move vision sensor to tip of Sawyer, tip is parent so from now on can use sim_handle_parent as relativeHandle
vrep.simSetObjectPosition(sawyer_vision_sensor, -1, tip_position)
sawyer_vision_sensor_position = vrep.simGetObjectPosition(sawyer_vision_sensor, -1)
print(sawyer_vision_sensor_position)

path = sawyer.get_path(position=target_cube_position, euler=[0.0, -np.pi, 0.0])
print(path)
tips, images = simulate_path(pr, path, sawyer)
save_images(images)


pr.stop()
pr.shutdown()
