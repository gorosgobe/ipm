import collections

import numpy as np
from pyrep.backend import sim
from pyrep.objects.dummy import Dummy
from pyrep.robots.arms.sawyer import Sawyer
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper

from simulation.camera import WristCamera


class SawyerRobot(object):
    SAWYER = "Sawyer"
    JOINT1 = "Sawyer_joint1"
    JOINT2 = "Sawyer_joint2"
    JOINT3 = "Sawyer_joint3"
    JOINT4 = "Sawyer_joint4"
    JOINT5 = "Sawyer_joint5"
    JOINT6 = "Sawyer_joint6"
    JOINT7 = "Sawyer_joint7"
    JOINTS = [
        JOINT1, JOINT2, JOINT3, JOINT4, JOINT5, JOINT6, JOINT7
    ]
    # looking downwards from initial position upwards
    LOOK_DOWNWARDS_POSITION = {
        2: -1.45,
        4: 1.4,
        #6: 1.57
        6: 1.65
    }

    def __init__(self, pr, move_to_initial_position=True, initial_close_gripper=False):
        """
        pr: PyRep instance, scene must have been launched and simulation 
        must have been started
        """
        self.pr = pr
        self.sawyer = Sawyer()
        self.gripper = BaxterGripper()
        self.tip_velocities = []
        self.sawyer_handle = sim.simGetObjectHandle(SawyerRobot.SAWYER)
        # Handles for every joint
        self.joint_handles = [sim.simGetObjectHandle(joint_name) for joint_name in SawyerRobot.JOINTS]

        if initial_close_gripper:
            self.close_gripper()
        # Move robot to initial angle position in simulation
        if move_to_initial_position:
            self.set_initial_angles()

        # Initial angles for all robot joints. If move_to_initial_position, these will be the ones after
        # moving the robot
        self.joint_initial_state = self.get_joint_angles()

        # save initial position and orientation of robot, after the initial position move
        self.absolute_position = sim.simGetObjectPosition(self.sawyer_handle, -1)
        self.absolute_orientation = sim.simGetObjectOrientation(self.sawyer_handle, -1)

    def get_joint_angles(self):
        return {
            joint_number + 1: sim.simGetJointPosition(joint_handle) \
            for joint_number, joint_handle in enumerate(self.joint_handles)
        }

    def actuate_gripper(self, pos):
        done = False
        while not done:
            done = self.gripper.actuate(pos, 0.1)
            self.pr.step()

    def open_gripper(self):
        self.actuate_gripper(1.0)

    def close_gripper(self):
        self.actuate_gripper(0.0)

    def grasp(self, obj):
        self.close_gripper()
        self.gripper.grasp(obj)

    @staticmethod
    def move_joint(joint_handle, target_angle):
        sim.simSetJointTargetPosition(joint_handle, target_angle)

    def move_sawyer_to(self, position, orientation):
        sim.simSetObjectPosition(self.sawyer_handle, -1, position)
        sim.simSetObjectOrientation(self.sawyer_handle, -1, orientation)

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
        count = 0
        while not done:
            prev_pos = np.array(self.sawyer.get_tip().get_position())
            self.pr.step()
            count += 1
            curr_pos = np.array(self.sawyer.get_tip().get_position())
            done = done or np.linalg.norm(curr_pos - prev_pos) < 1e-4 or count == 100

    def get_joint_velocities(self):
        return np.asarray(self.sawyer.get_joint_velocities())

    def _simulate_path(self, path):
        path.visualize()
        path.set_to_start()
        self.pr.step()

        tip_positions = []
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
            # print("JACOBIAN TIP VELOCITY", tip_velocity)
            dummy_object_velocity = sim.simGetObjectVelocity(self.sawyer.get_tip().get_handle())
            # print("DUMMY VELOCITY", dummy_object_velocity)
            tip_velocities.append(dummy_object_velocity[0])
            images.append(self.camera.get_image())

        return tip_positions, tip_velocities, images

    @staticmethod
    def add_angles(angles, offset):
        offset_complete_dict = collections.defaultdict(int, offset)
        # angles contains all joint angles (1 to 7)
        return {jn: angles[jn] + offset_complete_dict[jn] for jn in angles}

    def move_to_pos_and_orient(self, pos, orient):
        angles = self.sawyer.solve_ik(position=list(pos), euler=orient)
        self.set_angles({joint_number + 1: angle for joint_number, angle in enumerate(angles)})

    def move_along_velocity_and_orientation(self, tip_velocity, orientation):
        step = sim.simGetSimulationTimeStep()
        end_position = np.array(self.sawyer.get_tip().get_position()) + np.array(tip_velocity) * step
        self.move_to_pos_and_orient(end_position, orientation)

    def run_controller_simulation(self, controller, offset_angles={}, move_to_start_angles=True, target_distance=0.01):
        """
        Executes a simulation, starting from the default position, in a similar way to
        generate_image_simulation. However, in this case, the controller is used to obtain the tip velocity
        that needs to be applied given an image. 
        """
        if move_to_start_angles:
            self.set_initial_angles()

        self.set_angles(SawyerRobot.add_angles(self.joint_initial_state, offset_angles))
        images = []
        done = False
        count = 0
        achieved = False
        min_distance = None
        while not done:
            image = self.camera.get_image()
            images.append(image)
            # unnormalized image at original resolution, controller takes care of it
            tip_velocity = controller.get_tip_velocity(image)
            print("Count", count)
            print("Tip velocity: ", tip_velocity)
            step = sim.simGetSimulationTimeStep()
            end_position = np.array(self.sawyer.get_tip().get_position()) + np.array(tip_velocity) * step
            angles = self.sawyer.solve_ik(position=list(end_position), euler=[0.0, 0.0, 0.0])
            self.set_angles({joint_number + 1: angle for joint_number, angle in enumerate(angles)})
            count += 1
            dist = np.linalg.norm(
                np.array(self.sawyer.get_tip().get_position()) - self.get_target_reach_cube_position())
            print("Dist to target cube", dist)
            if dist < target_distance:
                achieved = True

            min_distance = min(dist, min_distance) if min_distance is not None else dist
            done = done or dist < target_distance or count == 100

        return images, achieved, min_distance

    def generate_image_simulation(
            self,
            offset={},
            move_to_start_angles=True,
            tip_velocities_zero_end=True
    ):
        """
        tip_velocities_zero_end: True if we want the last tip velocity to be zero, otherwise it is 
        set to the last tip velocity obtained. This is because in our model, we need to determine when to 
        stop the controller, and this occurs in the last image of the demonstration
        """

        if move_to_start_angles:
            # move all angles back to original position, as specified by
            # moving to default position and then to looking downwards position
            self.set_angles(self.joint_initial_state)

        self.set_angles(SawyerRobot.add_angles(self.joint_initial_state, offset))

        target_cube_position_above = self.get_target_reach_cube_position()
        path = self.sawyer.get_path(position=list(target_cube_position_above), euler=[0.0, 0.0, 0.0])
        tip_positions, tip_velocities, images = self._simulate_path(path)
        if tip_velocities_zero_end:
            tip_velocities[-1] = [0.0, 0.0, 0.0]
        return tip_positions, tip_velocities, images

    def get_target_reach_cube_position(self):
        # 5 cm above target cube is what we want
        target_cube_position = sim.simGetObjectPosition(self.target_cube_handle, -1)
        return np.array(target_cube_position) + np.array([0.0, 0.0, 0.05])

    @staticmethod
    def generate_offset():
        offset_angles_list = np.random.uniform(-np.pi / 30, np.pi / 30, size=7)
        offset_angles = {idx + 1: angle_offset for idx, angle_offset in enumerate(offset_angles_list)}
        return offset_angles
