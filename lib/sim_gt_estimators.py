import numpy as np
from pyrep.backend import sim


class SimGTEstimator(object):
    def __init__(self):
        self.step = sim.simGetSimulationTimeStep()
        self.stop_simulation = False

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

        if distance_to_target < 1e-4:
            self.stop_simulation = True
            return [0.0, 0.0, 0.0]

        difference_orientation = self.target_orientation - np.array(camera_orientation) + np.pi
        difference_orientation = (difference_orientation % (2 * np.pi)) - np.pi
        difference_orientation_normalised = difference_orientation / distance_to_target

        return difference_orientation_normalised


class SimGTVelocityEstimator(SimGTEstimator):
    def __init__(self, target_position):
        super().__init__()
        self.target_position = target_position

    def get_gt_tip_velocity(self, camera_position):
        distance_vector = np.array(self.target_position) - np.array(camera_position)
        distance_to_target = np.linalg.norm(distance_vector)

        if distance_to_target < 1e-4:
            self.stop_simulation = True
            return [0.0, 0.0, 0.0]

        return self.get_normalised_velocity(distance_vector, distance_to_target)

    def get_normalised_velocity(self, distance_vector, distance_vector_norm):
        # normalise, or the minimum when close enough
        if distance_vector_norm >= self.step:
            # if too big, normalise, velocity * step when applied
            velocity = distance_vector / distance_vector_norm
        else:
            # smaller than step, make it so it reaches target in one step
            velocity = distance_vector / self.step
        return velocity
