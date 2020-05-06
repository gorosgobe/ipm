import numpy as np
from pyrep.backend import sim


class SimGTEstimator(object):
    def __init__(self):
        self.step = sim.simGetSimulationTimeStep()
        self.stop_simulation = False
        self.precision = 2e-3
        self.switch_decrease = self.step + 0.06
        self.passed_discontinuity = False

    def stop_sim(self):
        return self.stop_simulation


class SimGTOrientationEstimator(SimGTEstimator):
    def __init__(self, target_position, target_orientation):
        super().__init__()
        self.target_position = target_position
        self.target_orientation = target_orientation

    def get_gt_orientation_change(self, camera_position, camera_orientation):
        distance_vector = np.array(self.target_position) - np.array(camera_position)
        distance_to_target = np.linalg.norm(distance_vector)

        if distance_to_target < self.precision:
            self.stop_simulation = True
            return np.array([0.0, 0.0, 0.0])

        difference_orientation = self.target_orientation - np.array(camera_orientation) + np.pi
        difference_orientation = (difference_orientation % (2 * np.pi)) - np.pi
        if distance_to_target < self.switch_decrease:
            distance_to_target = distance_to_target / (self.step * 3)
        difference_orientation_normalised = difference_orientation / distance_to_target
        return difference_orientation_normalised


class SimGTVelocityEstimator(SimGTEstimator):
    def __init__(self, target_position, generating=False, discontinuity=True):
        super().__init__()
        self.target_position = target_position
        self.generating = generating
        # do we want a discontinuity around the target position? orientation estimator is not affected by this
        self.discontinuity = discontinuity
        self.initial_direction = None

    def get_gt_tip_velocity(self, camera_position):
        distance_vector = np.array(self.target_position) - np.array(camera_position)
        # record initial direction at beginning of trajectory - this is used to determine the discontinuity jump
        # direction, so there is a wide variety of jumps across the dataset
        self.initial_direction = distance_vector if self.initial_direction is None else self.initial_direction
        distance_to_target = np.linalg.norm(distance_vector)
        should_zero = False

        if distance_to_target < self.precision:
            # if we have reached the target, we still want examples past the target
            # if we are simply evaluating the precision, we do not want jumps, but the precise answer
            # if we are not interested in a discontinuity, always hit this case
            if (not self.generating or self.passed_discontinuity) or not self.discontinuity:
                self.stop_simulation = True
                return np.array([0.0, 0.0, 0.0]), should_zero

            # one example past the target, to deal with discontinuity
            self.passed_discontinuity = True
            distance_jump = np.random.uniform(0.03, 0.05)
            # make velocity vector, with same direction as initial one, but size the sampled one
            sampled_cm_vec = self.initial_direction * distance_jump / np.linalg.norm(self.initial_direction)
            return self.get_normalised_velocity(sampled_cm_vec, distance_jump), True

        return self.get_normalised_velocity(distance_vector, distance_to_target), should_zero

    def get_normalised_velocity(self, distance_vector, distance_vector_norm):
        # normalise, or when close enough take more samples
        if distance_vector_norm >= self.switch_decrease:
            # if too big, normalise, velocity * step when applied
            velocity = distance_vector / distance_vector_norm
        else:
            # smaller than step, take a few examples to deal with discontinuity
            # exponential division per step
            velocity = distance_vector / (self.step * 3)
        return velocity
