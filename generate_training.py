from pyrep import PyRep
from pyrep.backend import vrep
from pyrep.robots.arms.sawyer import Sawyer
from os.path import join, dirname, abspath

SCENE_FILE = join(dirname(abspath(__file__)), "scenes/sawyer_reach_cube.ttt")

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

path = sawyer.get_path(position=target_cube_position, euler=[0.0, 0.0, 0.0])
print(path)
path.visualize()
path.set_to_start()
pr.step()

done = False
while not done:
    path.step()
    pr.step()

pr.stop()
pr.shutdown()
