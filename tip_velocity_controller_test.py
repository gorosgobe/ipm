from controller import TipVelocityController
from sawyer_robot import SawyerRobot
from scenes import SawyerReachCubeScene
import numpy as np

if __name__ == "__main__":
    with SawyerReachCubeScene(headless=True) as pr:
        sawyer_robot = SawyerRobot(pr)
        controller = TipVelocityController("models/finalbest.pt")
        for _ in range(5):
            offset_angles_list = np.random.uniform(-np.pi / 20, np.pi / 20, size=7)
            offset_angles = {idx + 1: angle_offset for idx, angle_offset in enumerate(offset_angles_list)}
            sawyer_robot.run_controller_simulation(controller, offset_angles)
