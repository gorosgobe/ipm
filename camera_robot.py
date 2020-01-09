import time

import numpy as np
from pyrep.backend import sim

from camera import MovableCamera


class CameraRobot(object):

    STEPS_PER_TRAJECTORY = 20

    def __init__(self, pr, show_paths=False):
        self.pr = pr
        self.movable_camera = MovableCamera(show_paths=show_paths)

    @staticmethod
    def generate_offset():
        return np.random.uniform(-0.3, 0.3, size=3)

    def generate_image_simulation(self, offset, target):
        print("Target", target)
        self.pr.step()
        self.movable_camera.set_initial_offset_position(offset)
        self.pr.step()
        self.movable_camera.set_orientation(target)
        self.pr.step()
        camera_position = self.movable_camera.get_position()
        distance_vector = np.array(target) - np.array(camera_position)
        step = sim.simGetSimulationTimeStep()
        simulation_steps_in_time_unit = 1 / step
        # -1 step because we take an image at the beginning too
        velocity = distance_vector * (simulation_steps_in_time_unit / (self.STEPS_PER_TRAJECTORY - 1))
        tip_positions = []
        tip_velocities = []
        images = []
        for i in range(self.STEPS_PER_TRAJECTORY):
            self.pr.step()
            tip_positions.append(self.movable_camera.get_position())
            tip_velocities.append(velocity)
            images.append(self.movable_camera.get_image())
            self.movable_camera.move_along_velocity(velocity)

        tip_velocities[-1] = [0.0, 0.0, 0.0]
        return tip_positions, tip_velocities, images

    def run_controller_simulation(self, controller, offset, target_distance=0.01):
        pass
