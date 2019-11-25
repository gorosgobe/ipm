from controller import TipVelocityController
from sawyer_robot import SawyerRobot
from scenes import SawyerReachCubeScene

if __name__ == "__main__":

    with SawyerReachCubeScene(headless=False) as pr:
        sawyer_robot = SawyerRobot(pr)
        controller   = TipVelocityController("models/model1574637168.pt")
        sawyer_robot.run_controller_simulation(controller)
