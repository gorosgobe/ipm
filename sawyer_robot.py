import collections
import numpy as np
from pyrep import PyRep
from pyrep.backend import vrep
from pyrep.robots.arms.sawyer import Sawyer
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper

class WristCamera(object):
    VISION_SENSOR = "Sawyer_wristCamera"
    def __init__(self):
        self.vision_sensor_handle = vrep.simGetObjectHandle(WristCamera.VISION_SENSOR)
        self.resolution = vrep.simGetVisionSensorResolution(self.vision_sensor_handle)

    def get_image(self):
        return vrep.simGetVisionSensorImage(self.vision_sensor_handle, self.resolution)

class SawyerRobot(object):

    SAWYER        = "Sawyer"
    JOINT1        = "Sawyer_joint1"
    JOINT2        = "Sawyer_joint2"
    JOINT3        = "Sawyer_joint3"
    JOINT4        = "Sawyer_joint4"
    JOINT5        = "Sawyer_joint5"
    JOINT6        = "Sawyer_joint6"
    JOINT7        = "Sawyer_joint7"
    JOINTS        = [
        JOINT1, JOINT2, JOINT3, JOINT4, JOINT5, JOINT6, JOINT7
    ]
    TARGET_CUBE   = "target_cube"
    # looking downwards from initial position upwards
    LOOK_DOWNWARDS_POSITION = { 
        2: -1.45, 
        4: 1.4, 
        6: 1.57 
    }

    def __init__(self, pr, move_to_initial_position=True):
        """
        pr: PyRep instance, scene must have been launched and simulation 
        must have been started
        """
        self.pr      = pr
        self.sawyer  = Sawyer()
        self.gripper = BaxterGripper()
        self.tip_velocities = []
        self.sawyer_handle = vrep.simGetObjectHandle(SawyerRobot.SAWYER)
        self.camera = WristCamera()
        self.target_cube_handle = vrep.simGetObjectHandle(SawyerRobot.TARGET_CUBE)
        # Handles for every joint
        self.joint_handles = [vrep.simGetObjectHandle(joint_name) for joint_name in SawyerRobot.JOINTS]

        # Move robot to initial angle position in simulation
        if move_to_initial_position:
            self.set_initial_angles()

        # Initial angles for all robot joints. If move_to_initial_position, these will be the ones after
        # moving the robot
        self.joint_initial_state = {
            joint_number + 1: vrep.simGetJointPosition(joint_handle) for joint_number, joint_handle in enumerate(self.joint_handles)
        }

        # save initial position and orientation of robot, after the initial position move
        self.absolute_position    = vrep.simGetObjectPosition(self.sawyer_handle, -1)
        self.absolute_orientation = vrep.simGetObjectOrientation(self.sawyer_handle, -1)

    @staticmethod
    def move_joint(joint_handle, target_angle):
        vrep.simSetJointTargetPosition(joint_handle, target_angle)

    def move_sawyer_to(self, position, orientation):
        vrep.simSetObjectPosition(self.sawyer_handle, -1, position)
        vrep.simSetObjectOrientation(self.sawyer_handle, -1, orientation)

    def get_joint_handle(self, joint_number):
        """
        joint_number: int, ranging from 1 to 7
        """
        if joint_number not in range(1, 8):
            raise Exception("Joint number should be between 1 and 7")
        return self.joint_handles[joint_number - 1]

    def set_initial_angles(self):
        self.set_angles(SawyerRobot.LOOK_DOWNWARDS_POSITION)

    def set_angles(self, angles):
        """
        angles: dictionary of (joint_number, angle) mappings
        """
        for joint_number in angles:
            SawyerRobot.move_joint(self.get_joint_handle(joint_number), angles[joint_number])

        done = False
        while not done:
            prev_pos = np.array(self.sawyer.get_tip().get_position())
            self.pr.step()
            curr_pos = np.array(self.sawyer.get_tip().get_position())
            done = done or np.linalg.norm(curr_pos - prev_pos) < 1e-4

    def get_joint_velocities(self):
        joint_velocities = [joint.get_joint_velocity() for joint in self.sawyer.joints]
        return np.asarray(joint_velocities)

    def _simulate_path(self, path):
        path.visualize()
        path.set_to_start()
        self.pr.step()

        tip_positions  = []
        tip_velocities = []
        images = []
        done = False
        while not done:
            done = path.step()
            self.pr.step()
            tip_positions.append(self.sawyer.get_tip().get_position())
            jacobian = self.sawyer.get_jacobian()
            joint_velocities = self.get_joint_velocities()
            tip_velocity = jacobian.T.dot(joint_velocities)
            tip_velocities.append(tip_velocity)
            images.append(self.camera.get_image())

        return tip_positions, tip_velocities, images

    @staticmethod
    def add_angles(angles, offset):
        offset_complete_dict = collections.defaultdict(int, offset)
        # angles contains all joint angles (1 to 7)
        return { jn: angles[jn] + offset_complete_dict[jn] for jn in angles }

    def run_simulation(
            self,
            offset_angles={},
            move_to_start_angles=True
        ):

        if move_to_start_angles:
            # move all angles back to original position, as specified by
            # moving to default position and then to looking downwards position
            self.set_angles(self.joint_initial_state)

        self.set_angles(SawyerRobot.add_angles(self.joint_initial_state, offset_angles))

        target_cube_position = vrep.simGetObjectPosition(self.target_cube_handle, -1)
        target_cube_position_above = np.array(target_cube_position) + np.array([0.0, 0.0, 0.08])
        path = self.sawyer.get_path(position=list(target_cube_position_above), euler=[0.0, 0.0, 0.0])

        return self._simulate_path(path)