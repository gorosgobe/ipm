import numpy as np
import torch
from pyrep.backend import sim

from lib.camera import MovableCamera


class CameraRobot(object):
    TEST_STEPS_PER_TRAJECTORY = 40
    TARGET_ORIENTATION = [-np.pi, 0, -np.pi / 2]

    def __init__(self, pr, show_paths=False):
        self.pr = pr
        self.movable_camera = MovableCamera(show_paths=show_paths)

    def get_movable_camera(self):
        return self.movable_camera

    @staticmethod
    def generate_offset():
        position_offset = np.random.uniform(-0.3, 0.3, size=3)
        # pi / 30 -> max of 9 degrees per axis in both directions
        # TODO: consider if rotation along z axis should be larger for higher variability
        # TODO: maybe consider this only if scene needs to be made more complex, before applying offset
        orientation_offset = np.random.uniform(-np.pi / 15, np.pi / 15, size=3)
        return np.concatenate((position_offset, orientation_offset), axis=0)

    def generate_image_simulation(self, offset, target_position, target_object, draw_center_pixel=False, debug=False):

        # position and orientation in 6 x 1 vector
        offset_position, offset_orientation = np.split(offset, 2)
        self.movable_camera.set_initial_offset_position(offset_position)
        self.movable_camera.add_to_orientation(offset_orientation)
        self.pr.step()

        target_handle = target_object.get_handle()

        tip_positions = []
        tip_velocities = []
        rotations = []
        relative_target_positions = []
        relative_target_orientations = []
        images = []
        crop_pixels = []
        while True:
            # world camera position
            camera_position = self.movable_camera.get_position()
            tip_positions.append(camera_position)

            # target position relative to camera
            relative_target_position = target_object.get_position(relative_to=self.movable_camera)
            print("Rel target position", relative_target_position)
            relative_target_positions.append(relative_target_position)

            # target orientation relative to camera
            relative_target_orientation = target_object.get_orientation(relative_to=self.movable_camera)
            print("Rel target orientation", relative_target_orientation)
            relative_target_orientations.append(relative_target_orientation)

            # world distance vector
            distance_vector = np.array(target_position) - np.array(camera_position)
            distance_vector_norm = np.linalg.norm(distance_vector)

            camera_orientation = np.array(self.movable_camera.get_orientation())
            print(camera_orientation)
            # Calculate difference in orientation and normalise over distance to target
            difference_orientation = self.TARGET_ORIENTATION - camera_orientation + np.pi
            difference_orientation = (difference_orientation % (2 * np.pi)) - np.pi
            difference_orientation_normalised = difference_orientation / np.linalg.norm(distance_vector_norm)

            step = sim.simGetSimulationTimeStep()
            # get pixel and extra information
            # TODO: refactor this to use object that knows how to compute pixel position from relative position, and pass
            # TODO: directly relative position from handle
            pixel, axis = self.movable_camera.compute_pixel_position(target_handle, debug=debug)
            crop_pixels.append(pixel)

            # get image, add debug info and record
            image = self.movable_camera.get_image()
            self.add_debug_info_to_img(axis, debug, draw_center_pixel, image, pixel)
            images.append(image)

            print("Dist to target at image taking time: ", np.linalg.norm(np.array(target_position) - np.array(self.movable_camera.get_position())))

            if distance_vector_norm < 0.0001:
                tip_velocities.append([0.0, 0.0, 0.0])
                rotations.append([0.0, 0.0, 0.0])
                break
            else:
                velocity = self.get_normalised_velocity(distance_vector, distance_vector_norm)
                tip_velocities.append(velocity)
                rotations.append(difference_orientation_normalised)

            self.movable_camera.add_to_orientation(step * difference_orientation_normalised)
            self.movable_camera.move_along_velocity(velocity)
            self.pr.step()

        return dict(
            tip_positions=tip_positions,
            tip_velocities=tip_velocities,
            images=images,
            crop_pixels=crop_pixels,
            rotations=rotations,
            relative_target_positions=relative_target_positions,
            relative_target_orientations=relative_target_orientations
        )


    @staticmethod
    def get_normalised_velocity(distance_vector, distance_vector_norm):
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

    def run_controller_simulation(self, controller, offset, target, target_distance=0.01, has_rotations=True):
        # target is np array
        # TODO: remove repetition with generate_image_simulation
        offset_position, offset_orientation = np.split(offset, 2)
        self.movable_camera.set_initial_offset_position(offset_position)
        self.movable_camera.add_to_orientation(offset_orientation)
        self.pr.step()

        achieved = False
        images = []
        tip_velocities = []
        rotations = []
        min_distance = None
        step = sim.simGetSimulationTimeStep()
        for i in range(self.TEST_STEPS_PER_TRAJECTORY):
            image = self.movable_camera.get_image()
            images.append(image)

            dist = np.linalg.norm(target - np.array(self.movable_camera.get_position()))
            min_distance = dist if min_distance is None else min(min_distance, dist)
            if dist < target_distance:
                achieved = True
                break

            control = np.array(controller.get_tip_control(image))

            # apply rotation
            if has_rotations:
                velocity, rotation = np.split(control, 2)
                rotations.append(rotation)
                self.movable_camera.add_to_orientation(step * rotation)
            else:
                velocity = control

            # apply velocity
            tip_velocities.append(velocity)
            self.movable_camera.move_along_velocity(velocity)
            self.pr.step()

        return images, tip_velocities, achieved, min_distance
