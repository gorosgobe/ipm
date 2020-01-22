import numpy as np
from pyrep.backend import sim

from lib.camera import MovableCamera


class CameraRobot(object):
    TEST_STEPS_PER_TRAJECTORY = 40

    def __init__(self, pr, show_paths=False):
        self.pr = pr
        self.movable_camera = MovableCamera(show_paths=show_paths)

    def get_movable_camera(self):
        return self.movable_camera

    @staticmethod
    def generate_offset():
        position_offset = np.random.uniform(-0.3, 0.3, size=3)
        # pi / 30 -> max of 6 degrees per axis
        orientation_offset = np.random.uniform(-np.pi / 30, np.pi / 30, size=3)
        return np.concatenate((position_offset, orientation_offset), axis=0)

    def generate_image_simulation(self, offset, target, target_object, draw_center_pixel=False, debug=False):
        # position and orientation in 6 x 1 vector
        offset_position, offset_orientation = np.split(offset, 2)
        self.movable_camera.set_initial_offset_position(offset_position)
        # TODO: fix this
        self.movable_camera.set_orientation(self.movable_camera.get_orientation() + offset_orientation)
        # self.movable_camera.set_orientation_towards(target)
        self.pr.step()

        target_handle = target_object.get_handle()

        tip_positions = []
        tip_velocities = []
        images = []
        crop_pixels = []
        while True:
            camera_position = self.movable_camera.get_position()
            tip_positions.append(camera_position)

            distance_vector = np.array(target) - np.array(camera_position)
            distance_vector_norm = np.linalg.norm(distance_vector)
            if distance_vector_norm < 0.0001:
                break

            velocity = self.normalise_velocity(distance_vector, distance_vector_norm)
            tip_velocities.append(velocity)

            # get pixel and extra information
            # TODO: refactor this to use object that knows how to compute pixel position from relative position, and pass
            # TODO: directly relative position from handle
            pixel, axis = self.movable_camera.compute_pixel_position(target_handle, debug=debug)
            crop_pixels.append(pixel)

            # get image, add debug info and record
            image = self.movable_camera.get_image()
            self.add_debug_info_to_img(axis, debug, draw_center_pixel, image, pixel)
            images.append(image)

            self.movable_camera.move_along_velocity(velocity)
            self.pr.step()
            print("Dist to target", np.linalg.norm(np.array(target) - np.array(self.movable_camera.get_position())))

        tip_velocities[-1] = [0.0, 0.0, 0.0]
        return tip_positions, tip_velocities, images, crop_pixels

    @staticmethod
    def normalise_velocity(distance_vector, distance_vector_norm):
        step = sim.simGetSimulationTimeStep()
        # normalise, or the minimum when close enough
        if distance_vector_norm >= step:
            # if too big, normalise, velocity * step when applied
            velocity = distance_vector / distance_vector_norm
        else:
            # smaller than step, make it so it reaches target in one step
            velocity = distance_vector / step
        return velocity

    @staticmethod
    def add_debug_info_to_img(axis, debug, draw_center_pixel, image, pixel):
        if draw_center_pixel:
            image[pixel[1], pixel[0]] = np.array([1.0, 1.0, 0.0])
        if debug:
            colours = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            for idx, ax in enumerate(axis):
                for p in ax:
                    px, py = p
                    image[py, px] = np.array(colours[idx])

    def run_controller_simulation(self, controller, offset, target, target_distance=0.01):
        # target is np array
        self.movable_camera.set_initial_offset_position(offset)
        self.movable_camera.set_orientation(target)
        self.pr.step()

        achieved = False
        images = []
        tip_velocities = []
        min_distance = None
        for i in range(self.TEST_STEPS_PER_TRAJECTORY):
            image = self.movable_camera.get_image()
            images.append(image)
            velocity = np.array(controller.get_tip_velocity(image))
            tip_velocities.append(velocity)
            self.movable_camera.move_along_velocity(velocity)
            self.pr.step()
            dist = np.linalg.norm(target - np.array(self.movable_camera.get_position()))
            min_distance = dist if min_distance is None else min(min_distance, dist)
            if dist < target_distance:
                achieved = True
                break

        return images, tip_velocities, achieved, min_distance
