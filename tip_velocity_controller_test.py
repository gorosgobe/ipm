from controller import TipVelocityController, CenterCropper
from sawyer_robot import SawyerRobot
from scenes import SawyerReachCubeScene
import numpy as np

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull

if __name__ == "__main__":

    @contextmanager
    def suppress_stdout_stderr():
        """A context manager that redirects stdout and stderr to devnull"""
        with open(devnull, 'w') as fnull:
            with redirect_stderr(fnull) as err:
                yield err

    with SawyerReachCubeScene(headless=False) as pr:
        sawyer_robot = SawyerRobot(pr)
        controller = TipVelocityController("models/M1_40_0.002699121600613953.pt", CenterCropper(640 // 6, 480 // 6))
        print(controller.get_model().test_loss)

        for _ in range(5):
            offset_angles_list = np.random.uniform(-np.pi / 30, np.pi / 30, size=7)
            offset_angles = {idx + 1: angle_offset for idx, angle_offset in enumerate(offset_angles_list)}
            sawyer_robot.run_controller_simulation(controller, offset_angles)
