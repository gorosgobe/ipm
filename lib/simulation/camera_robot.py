import numpy as np
from pyrep.backend import sim
from pyrep.objects.shape import Shape

from simulation.camera import MovableCamera
from simulation.sim_gt_estimators import SimGTVelocityEstimator, SimGTOrientationEstimator


class CameraRobot(object):
    TEST_STEPS_PER_TRAJECTORY = 40

    def __init__(self, pr, show_paths=False, movable=None, offset_generator=None):
        self.pr = pr
        self.movable_camera = movable or MovableCamera(show_paths=show_paths)
        self.offset_generator = offset_generator

    def get_movable_camera(self):
        return self.movable_camera

    def generate_offset(self):
        if self.offset_generator is not None:
            return self.offset_generator.generate_offset()
        xy_position_offset = np.random.uniform(-0.4, 0.4, size=2)
        # otherwise, sometimes camera is too far away
        z_position_offset = np.random.uniform(-0.3, 0.3, size=1)
        xy_orientation_offset = np.random.uniform(-np.pi / 15, np.pi / 15, size=2)
        # increased initial rotation for z-axis
        z_offset = np.random.normal(0, np.pi / 5, size=1)
        return np.concatenate((xy_position_offset, z_position_offset, xy_orientation_offset, z_offset), axis=0)

    def generate_image_simulation(self, offset, scene, target_position, target_object, sim_gt_velocity, sim_gt_orientation,
                                  draw_center_pixel=False, debug=False, randomise_distractors=False):

        # position and orientation in 6 x 1 vector
        offset_position, offset_orientation = np.split(offset, 2)
        self.movable_camera.set_initial_offset_position_and_orientation(offset_position, offset_orientation)

        distractors = scene.get_distractors()
        dsp = scene.get_distractor_safe_distances()
        if randomise_distractors:
            self.set_distractor_random_positions(target_position, distractors, dsp)

        distractor_positions = [d.get_position() for d in distractors]

        self.pr.step()

        tip_positions = []
        tip_velocities = []
        rotations = []
        relative_target_positions = []
        relative_target_orientations = []
        images = []
        crop_pixels = []
        count = 0
        count_stop_demonstration = None
        while True:
            count += 1
            # world camera/tip position
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
            pixel, axis = self.movable_camera.compute_pixel_position(target_object.get_handle(), debug=debug)
            crop_pixels.append(pixel)

            # get image, add debug info and record
            image = self.movable_camera.get_image()
            self.add_debug_info_to_img(axis, debug, draw_center_pixel, image, pixel)
            images.append(image)

            print("Dist to target at image taking time: ",
                  np.linalg.norm(np.array(target_position) - np.array(camera_position)))

            velocity, should_zero = sim_gt_velocity.get_gt_tip_velocity(camera_position)
            print("velocity", velocity)
            if should_zero:
                # number of images in demonstration before discontinuity
                count_stop_demonstration = count
            # if should_zero, apply velocity but append zeros so model does not get confused
            # this is the case when we want to add more data around target discontinuity
            tip_velocities.append(velocity if not should_zero else np.zeros(3))
            difference_orientation_normalised = sim_gt_orientation.get_gt_orientation_change(camera_position,
                                                                                             camera_orientation)
            print("rotation", difference_orientation_normalised)
            rotations.append(difference_orientation_normalised)
            if sim_gt_velocity.stop_sim() and sim_gt_orientation.stop_sim():
                break

            self.movable_camera.move_along_velocity_and_add_orientation(velocity, step * difference_orientation_normalised)
            self.pr.step()

        return dict(
            tip_positions=tip_positions,
            tip_velocities=tip_velocities,
            images=images,
            crop_pixels=crop_pixels,
            rotations=rotations,
            relative_target_positions=relative_target_positions,
            relative_target_orientations=relative_target_orientations,
            distractor_positions=distractor_positions,
            count_stop_demonstration=count_stop_demonstration
        )

    @staticmethod
    def get_x_distractor():
        return np.random.uniform(-0.3, 1.2)

    @staticmethod
    def get_y_distractor():
        return np.random.uniform(-0.7, 0.8)

    @staticmethod
    def set_distractor_random_positions(target_position, distractors, dsp):
        x_target = target_position[0]
        y_target = target_position[1]
        target = np.array([x_target, y_target])
        try:
            robot = Shape("Sawyer").get_position()[:-1]
        except:
            robot = target
        distractor_positions = []
        for idx, d in enumerate(distractors):
            previous_distractors = distractors[:idx]
            # get random position within table dimensions
            x = target[0]
            y = target[1]
            # make sure obtained x is not within safe distance of target or previously set distractors
            while np.linalg.norm(np.array([x, y]) - target) < dsp[idx] or \
                    np.linalg.norm(np.array([x, y]) - robot) < dsp[idx] or \
                    any(filter(lambda other_d: np.linalg.norm(np.array([x, y]) - np.array(other_d.get_position()[:2])) < dsp[idx], previous_distractors)):
                x = CameraRobot.get_x_distractor()
                y = CameraRobot.get_y_distractor()

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

    def run_controller_simulation(self, controller, offset, target_position, scene, sim_gt_velocity, sim_gt_orientation, target_distance=0.01,
                                  fixed_steps=-1, distractor_positions=None, break_early=False):
        # target is np array
        # set camera position and orientation
        offset_position, offset_orientation = np.split(offset, 2)
        self.movable_camera.set_initial_offset_position_and_orientation(offset_position, offset_orientation)
        # set distractor object positions
        dists = scene.get_distractors()
        if distractor_positions is not None:
            # scene 1 case
            for idx, d in enumerate(dists):
                d.set_position(distractor_positions[idx])
        self.pr.step()

        achieved = False
        images = []
        tip_velocities = []
        rotations = []
        min_distance = None
        # not generating
        combined_errors = []
        velocity_errors = []
        orientation_errors = []
        step = sim.simGetSimulationTimeStep()
        fixed_steps_distance = -1
        # run default test steps, unless a fixed number of steps have to be run (i.e. when dataset is padded)
        for i in range(self.TEST_STEPS_PER_TRAJECTORY if fixed_steps == -1 else fixed_steps + 1):
            image = self.movable_camera.get_image()
            images.append(image)
            camera_position = self.movable_camera.get_position()
            dist = np.linalg.norm(target_position - np.array(camera_position))
            min_distance = dist if min_distance is None else min(min_distance, dist)

            # when fixed steps are not taken into account, break when target distance is reached
            if fixed_steps == -1:
                if dist < target_distance:
                    achieved = True
                    break
            else:
                fixed_steps_distance = dist
                if break_early:
                    # disc insertion scene
                    xy = np.array(camera_position)[:-1]
                    target_xy = target_position[:-1]
                    # radius of peg is 1.5cm, so disc center should be in that region
                    # height of tip position should be within the peg, end height in all demonstrations is 0.62
                    # but its okay to insert it fully
                    print("Norm", np.linalg.norm(target_xy - xy))
                    print("Camera height", camera_position[-1])
                    if np.linalg.norm(target_xy - xy) < 0.015 and 0.6 <= camera_position[-1] < 0.63:
                        achieved = True
                        break
                        # disc is outside of peg in xy (centers not close enough given size of disc) and below start of peg, so trajectory fails
                    elif np.linalg.norm(target_xy - xy) > 0.01 and camera_position[-1] < 0.65:
                        break
                if i == fixed_steps:
                    # test episode has ended
                    break

            control = np.array(controller.get_tip_control(image))
            gt_velocity, _ = sim_gt_velocity.get_gt_tip_velocity(camera_position)
            gt_velocity = np.array(gt_velocity)
            camera_orientation = self.movable_camera.get_orientation()
            gt_orientation = np.array(sim_gt_orientation.get_gt_orientation_change(camera_position, camera_orientation))
            combined_gt = np.concatenate((gt_velocity, gt_orientation), axis=0)
            # total error
            combined_error_norm = np.linalg.norm(combined_gt - control)
            combined_errors.append(
                dict(error_norm=combined_error_norm, gt=combined_gt.tolist(), predicted=control.tolist()))
            velocity, rotation = np.split(control, 2)
            # velocity error
            velocity_error_norm = np.linalg.norm(gt_velocity - velocity)
            velocity_errors.append(
                dict(error_norm=velocity_error_norm, gt=gt_velocity.tolist(), predicted=velocity.tolist()))
            # rotation error
            rotation_error_norm = np.linalg.norm(gt_orientation - rotation)
            orientation_errors.append(
                dict(error_norm=rotation_error_norm, gt=gt_orientation.tolist(), predicted=rotation.tolist()))
            print("Combined error", combined_error_norm, "vel", velocity_error_norm, "rot", rotation_error_norm)

            rotations.append(rotation)
            tip_velocities.append(velocity)
            self.movable_camera.move_along_velocity_and_add_orientation(velocity, step * rotation)
            self.pr.step()

        assert fixed_steps_distance == -1 if fixed_steps == -1 else True

        return dict(
            images=images,
            tip_velocities=tip_velocities,
            achieved=achieved,
            min_distance=min_distance,
            fixed_steps_distance=fixed_steps_distance,
            combined_errors=combined_errors,
            velocity_errors=velocity_errors,
            orientation_errors=orientation_errors
        )
