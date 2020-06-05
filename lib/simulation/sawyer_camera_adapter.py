import numpy as np

from pyrep.objects.shape import Shape
from lib.simulation.camera import MovableCamera
from lib.simulation.sawyer_robot import SawyerRobot


class SawyerCameraAdapter(object):
    VISION_SENSOR = "Sawyer_wristCamera"
    DISC = "disc"

    def __init__(self, pr):
        self.sawyer_robot = SawyerRobot(pr, move_to_initial_position=True, initial_close_gripper=True)
        self.tip = self.sawyer_robot.sawyer.get_tip()
        self.initial_position = self.tip.get_position()
        self.initial_orientation = self.tip.get_orientation()
        self.movable_camera = MovableCamera(name=SawyerCameraAdapter.VISION_SENSOR)

    def set_initial_offset_position_and_orientation(self, offset_position, offset_orientation):
        self.sawyer_robot.set_initial_angles()
        self.sawyer_robot.move_to_pos_and_orient(
            pos=list(np.array(self.initial_position) + np.array(offset_position)),
            orient=list(np.array(self.initial_orientation) + np.array(offset_orientation))
        )

    def get_position(self):
        return self.tip.get_position()

    def get_orientation(self):
        return self.tip.get_orientation()

    def get_handle(self):
        return self.tip.get_handle()

    def compute_pixel_position(self, handle, debug=False):
        return self.movable_camera.compute_pixel_position(handle=handle, debug=debug)

    def get_image(self):
        return self.movable_camera.get_image()

    def move_along_velocity_and_add_orientation(self, vel, rot):
        orientation = self.tip.get_orientation()
        self.sawyer_robot.move_along_velocity_and_orientation(
            tip_velocity=vel,
            orientation=list(np.array(orientation) + rot)
        )
