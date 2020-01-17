import time

import numpy as np
from pyrep.backend import sim
from pyrep.objects.shape import Shape

from camera import MovableCamera


class CameraRobot(object):

    STEPS_PER_TRAJECTORY = 20

    def __init__(self, pr, show_paths=False):
        self.pr = pr
        self.movable_camera = MovableCamera(show_paths=show_paths)

    @staticmethod
    def generate_offset():
        return np.random.uniform(-0.3, 0.3, size=3)

    def generate_image_simulation(self, offset, target, target_object, draw_center_pixel=False, debug=False):
        self.movable_camera.set_initial_offset_position(offset)
        print("Before orientation", sim.simGetObjectPosition(target_object.get_handle(), self.movable_camera.vision_sensor_handle))
        self.movable_camera.set_orientation(target + np.array([0.2, 0.2, 0.0]))
        print("After orientation", sim.simGetObjectPosition(target_object.get_handle(), self.movable_camera.vision_sensor_handle))
        self.pr.step()

        target_handle = target_object.get_handle()
        camera_position = self.movable_camera.get_position()
        distance_vector = np.array(target) - np.array(camera_position)
        step = sim.simGetSimulationTimeStep()
        simulation_steps_in_time_unit = 1 / step
        velocity = distance_vector * (simulation_steps_in_time_unit / self.STEPS_PER_TRAJECTORY)
        tip_positions = []
        tip_velocities = []
        images = []
        crop_pixels = []
        for i in range(self.STEPS_PER_TRAJECTORY):
            tip_positions.append(self.movable_camera.get_position())
            tip_velocities.append(velocity)
            # estimate, is not very accurate
            pixel, axis = self.movable_camera.compute_pixel_position(target_handle, debug=debug)
            crop_pixels.append(pixel)
            image = self.movable_camera.get_image()
            if draw_center_pixel:
                image[pixel[1], pixel[0]] = np.array([1.0, 1.0, 0.0])
            if debug:
                colours = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                for idx, ax in enumerate(axis):
                    for p in ax:
                        px, py = p
                        image[py, px] = np.array(colours[idx])
            images.append(image)
            self.movable_camera.move_along_velocity(velocity)
            self.pr.step()
            print("Dist to target", np.linalg.norm(np.array(target) - np.array(self.movable_camera.get_position())))

        tip_velocities[-1] = [0.0, 0.0, 0.0]
        return tip_positions, tip_velocities, images, crop_pixels

    def run_controller_simulation(self, controller, offset, target, target_object, target_distance=0.01):
        self.movable_camera.set_initial_offset_position(offset)
        self.movable_camera.set_orientation(Shape("target_cube").get_position())
        self.pr.step()
        pass
