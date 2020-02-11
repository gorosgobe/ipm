import numpy as np
import torch
from pyrep.backend import sim

from lib import utils
from lib.camera import MovableCamera
from lib.sim_gt_estimators import SimGTVelocityEstimator, SimGTOrientationEstimator


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
        xy_position_offset = np.random.uniform(-0.4, 0.4, size=2)
        # otherwise, sometimes camera is too far away
        z_position_offset = np.random.uniform(-0.3, 0.3, size=1)
        xy_orientation_offset = np.random.uniform(-np.pi / 15, np.pi / 15, size=2)
        # increased initial rotation for z-axis
        z_offset = np.random.normal(0, np.pi / 5, size=1)
        return np.concatenate((xy_position_offset, z_position_offset, xy_orientation_offset, z_offset), axis=0)

    def generate_image_simulation(self, offset, scene, target_position, target_object, draw_center_pixel=False,
                                  debug=False,
                                  randomise_distractors=False):

        # position and orientation in 6 x 1 vector
        offset_position, offset_orientation = np.split(offset, 2)
        self.movable_camera.set_initial_offset_position(offset_position)
        self.movable_camera.add_to_orientation(offset_orientation)

        if randomise_distractors:
            self.set_distractor_random_positions(scene, target_position)

        distractor_positions = [d.get_position() for d in scene.get_distractors()]

        self.pr.step()

        target_handle = target_object.get_handle()

        tip_positions = []
        tip_velocities = []
        rotations = []
        relative_target_positions = []
        relative_target_orientations = []
        images = []
        crop_pixels = []
        sim_gt_velocity = SimGTVelocityEstimator(target_position)
        sim_gt_orientation = SimGTOrientationEstimator(target_position, self.TARGET_ORIENTATION)
        while True:
            # world camera position
            camera_position = self.movable_camera.get_position()
            tip_positions.append(camera_position)

            # target position relative to camera
            relative_target_position = target_object.get_position(relative_to=self.movable_camera)
            relative_target_positions.append(relative_target_position)

            # target orientation relative to camera
            relative_target_orientation = target_object.get_orientation(relative_to=self.movable_camera)
            relative_target_orientations.append(relative_target_orientation)

            camera_orientation = self.movable_camera.get_orientation()

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

            print("Dist to target at image taking time: ",
                  np.linalg.norm(np.array(target_position) - np.array(camera_position)))

            velocity = sim_gt_velocity.get_gt_tip_velocity(camera_position)
            tip_velocities.append(velocity)
            difference_orientation_normalised = sim_gt_orientation.get_gt_orientation_change(camera_position,
                                                                                             camera_orientation)
            rotations.append(difference_orientation_normalised)

            if sim_gt_velocity.stop_sim() and sim_gt_orientation.stop_sim():
                break

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
            relative_target_orientations=relative_target_orientations,
            distractor_positions=distractor_positions
        )

    @staticmethod
    def get_x_distractor():
        return np.random.uniform(-0.3, 1.2)

    @staticmethod
    def get_y_distractor():
        return np.random.uniform(-0.7, 0.8)

    def set_distractor_random_positions(self, scene, target_position):
        """
        Sets the positions of distractor objects randomly. Every distractor will be at least 10cm away from the target
        and other distractors.
        :param scene: The scene to get the distractors from
        :param target_position: The position of the target
        :return: The positions the distractors were set at
        """
        distractors = scene.get_distractors()
        dsp = scene.get_distractor_safe_distances()
        x_target = target_position[0]
        y_target = target_position[1]
        distractor_positions = []
        for idx, d in enumerate(distractors):
            previous_distractors = distractors[:idx]
            # get random position within table dimensions
            x = x_target
            y = y_target
            # make sure obtained x is not within safe distance of target or previously set distractors
            while abs(x - x_target) < dsp[idx] or \
                    any(filter(lambda other_d: abs(x - other_d.get_position()[0]) < dsp[idx], previous_distractors)):
                x = self.get_x_distractor()

            while abs(y - y_target) < dsp[idx] or \
                    any(filter(lambda other_d: abs(y - other_d.get_position()[1]) < dsp[idx], previous_distractors)):
                y = self.get_y_distractor()

            d_position = [x, y, d.get_position()[-1]]
            d.set_position(d_position)
            distractor_positions.append(d_position)

        return distractor_positions

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

    def run_controller_simulation(self, controller, offset, target, distractor_positions, scene, target_distance=0.01):
        # target is np array
        # TODO: remove repetition with generate_image_simulation
        # set camera position and orientation
        offset_position, offset_orientation = np.split(offset, 2)
        self.movable_camera.set_initial_offset_position(offset_position)
        self.movable_camera.add_to_orientation(offset_orientation)
        # set distractor object positions
        for idx, d in enumerate(scene.get_distractors()):
            d.set_position(distractor_positions[idx])
        self.pr.step()

        achieved = False
        images = []
        tip_velocities = []
        rotations = []
        min_distance = None
        sim_gt_velocity = SimGTVelocityEstimator(target)
        sim_gt_orientation = SimGTOrientationEstimator(target, self.TARGET_ORIENTATION)
        combined_errors = []
        velocity_errors = []
        orientation_errors = []
        step = sim.simGetSimulationTimeStep()
        for i in range(self.TEST_STEPS_PER_TRAJECTORY):
            image = self.movable_camera.get_image()
            images.append(image)
            camera_position = self.movable_camera.get_position()
            dist = np.linalg.norm(target - np.array(camera_position))
            min_distance = dist if min_distance is None else min(min_distance, dist)
            if dist < target_distance:
                achieved = True
                break
            control = np.array(controller.get_tip_control(image))
            gt_velocity = np.array(sim_gt_velocity.get_gt_tip_velocity(camera_position))
            camera_orientation = self.movable_camera.get_orientation()
            gt_orientation = np.array(sim_gt_orientation.get_gt_orientation_change(camera_position, camera_orientation))
            combined_gt = np.concatenate((gt_velocity, gt_orientation), axis=0)
            # total error
            combined_error_norm = np.linalg.norm(combined_gt - control)
            print("Combined error", combined_error_norm)
            combined_errors.append(dict(error_norm=combined_error_norm, gt=combined_gt.tolist(), predicted=control.tolist()))

            velocity, rotation = np.split(control, 2)
            # velocity error
            velocity_error_norm = np.linalg.norm(gt_velocity - velocity)
            velocity_errors.append(dict(error_norm=velocity_error_norm, gt=gt_velocity.tolist(), predicted=velocity.tolist()))
            # rotation error
            rotation_error_norm = np.linalg.norm(gt_orientation - rotation)
            orientation_errors.append(dict(error_norm=rotation_error_norm, gt=gt_orientation.tolist(), predicted=rotation.tolist()))

            rotations.append(rotation)
            # apply rotation
            self.movable_camera.add_to_orientation(step * rotation)
            # apply velocity
            tip_velocities.append(velocity)
            self.movable_camera.move_along_velocity(velocity)
            self.pr.step()

        return dict(
            images=images,
            tip_velocities=tip_velocities,
            achieved=achieved,
            min_distance=min_distance,
            combined_errors=combined_errors,
            velocity_errors=velocity_errors,
            orientation_errors=orientation_errors
        )
