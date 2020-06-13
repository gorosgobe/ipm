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
    def __init__(self, target_position, target_orientation, return_zero=True, reduce_factor=3, return_target=False):
        super().__init__()
        self.target_position = target_position
        self.target_orientation = target_orientation
        self.return_zero = return_zero
        self.reduce_factor = reduce_factor
        self.return_target = return_target

    def get_gt_orientation_change(self, camera_position, camera_orientation):
        distance_vector = np.array(self.target_position) - np.array(camera_position)
        distance_to_target = np.linalg.norm(distance_vector)

        if distance_to_target < self.precision:
            self.stop_simulation = True
            if self.return_zero:
                return np.array([0.0, 0.0, 0.0])

        if self.return_target:
            return np.array([0.0, 0.0, 0.0])

        difference_orientation = self.target_orientation - np.array(camera_orientation)
        difference_orientation = ((difference_orientation + np.pi) % (2 * np.pi)) - np.pi
        if distance_to_target < self.switch_decrease:
            distance_to_target = distance_to_target / (self.step * self.reduce_factor)
        difference_orientation_normalised = difference_orientation / distance_to_target
        return difference_orientation_normalised


class SimGTVelocityEstimator(SimGTEstimator):
    def __init__(self, target_position, generating=False, discontinuity=True, reduce_factor=3, return_zero=True,
                 slow_down_distance=None, slower=False):
        super().__init__()
        self.target_position = target_position
        self.generating = generating
        # do we want a discontinuity around the target position? orientation estimator is not affected by this
        self.discontinuity = discontinuity
        self.initial_direction = None
        self.reduce_factor = reduce_factor
        self.return_zero = return_zero
        self.slow_down_distance = slow_down_distance
        self.slower = slower

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
                if self.return_zero:
                    return np.array([0.0, 0.0, 0.0]), should_zero
                else:
                    return self.get_normalised_velocity(distance_vector, distance_to_target), should_zero

            # one example past the target, to deal with discontinuity
            self.passed_discontinuity = True
            distance_jump = np.random.uniform(0.03, 0.05)
            # make velocity vector, with same direction as initial one, but size the sampled one
            sampled_cm_vec = self.initial_direction * distance_jump / np.linalg.norm(self.initial_direction)
            return self.get_normalised_velocity(sampled_cm_vec, distance_jump), True

        return self.get_normalised_velocity(distance_vector, distance_to_target), should_zero

    def get_normalised_velocity(self, distance_vector, distance_vector_norm):
        # normalise, or when close enough take more samples
        if distance_vector_norm >= (self.slow_down_distance or self.switch_decrease):
            # if too big, normalise, velocity * step when applied
            velocity = distance_vector / distance_vector_norm
        else:
            # smaller than step, take a few examples to deal with discontinuity
            # exponential division per step
            velocity = distance_vector / (self.step * self.reduce_factor)

        if self.slower:
            velocity *= 0.5
        return velocity


class ContinuousGTEstimator(object):
    def __init__(self, waypoints, generating=False, reduce_factors=(3,)):
        self.waypoints = waypoints
        if len(reduce_factors) != len(waypoints):
            raise ValueError("Same waypoints as number of reduce factors!")
        self.velocity_estimators = [
            SimGTVelocityEstimator(
                way.get_position(),
                generating=generating,
                discontinuity=False,
                reduce_factor=reduce_factors[idx],
                return_zero=idx == len(waypoints) - 1,
                slower=True
            ) for idx, way in enumerate(waypoints)
        ]
        self.orientation_estimators = [
            SimGTOrientationEstimator(way.get_position(), [-np.pi, 0, -np.pi],
                                      return_target=True
                                      ) for idx, way in enumerate(waypoints)
        ]
        self.current_waypoint_idx = 0

    def increase_idx_if_waypoint_reached(self):
        if self.velocity_estimators[self.current_waypoint_idx].stop_sim() and \
                self.orientation_estimators[self.current_waypoint_idx].stop_sim():
            self.current_waypoint_idx += 1

    def get_gt_tip_velocity(self, tip_position):
        vel = self.velocity_estimators[self.current_waypoint_idx].get_gt_tip_velocity(tip_position)
        return vel

    def get_gt_orientation_change(self, tip_position, tip_orientation):
        res = self.orientation_estimators[self.current_waypoint_idx].get_gt_orientation_change(
            tip_position,
            tip_orientation
        )
        self.increase_idx_if_waypoint_reached()
        return res

    def stop_sim(self):
        return self.velocity_estimators[-1].stop_sim() and self.orientation_estimators[-1].stop_sim()
