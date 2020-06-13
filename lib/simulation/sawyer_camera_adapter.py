import numpy as np
from pyrep.objects.dummy import Dummy

from lib.simulation.camera import MovableCamera
from lib.simulation.sawyer_robot import SawyerRobot
from test_utils import downsample_coordinates
from utils import ResizeTransform


class SawyerCameraAdapter(object):
    VISION_SENSOR = "Sawyer_wristCamera"
    DISC = "disc"
    REMOVE_BOTTOM = 95

    def __init__(self, pr):
        self.sawyer_robot = SawyerRobot(pr, move_to_initial_position=True)
        self.tip = self.sawyer_robot.sawyer.get_tip()
        self.initial_position = self.tip.get_position()
        self.initial_orientation = self.tip.get_orientation()
        self.movable_camera = MovableCamera(name=SawyerCameraAdapter.VISION_SENSOR)
        self.pr = pr
        self.first = True

    def set_initial_offset_position_and_orientation(self, offset_position, offset_orientation):
        if not self.first:
            self.pr.stop()
            self.pr.start()
        self.first = False
        self.sawyer_robot.close_gripper()
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
        pixel, rest = self.movable_camera.compute_pixel_position(handle=handle, debug=debug)
        return downsample_coordinates(
            pixel[0],
            pixel[1],
            og_width=640,
            og_height=480-self.REMOVE_BOTTOM,
            to_width=640,
            to_height=480
        ), rest

    def get_image(self):
        image = self.movable_camera.get_image()
        # remove bottom part as this gives issues due to shadows rendered by openGL3
        return ResizeTransform((640, 480))(image[:-self.REMOVE_BOTTOM, :, :])

    def move_along_velocity_and_add_orientation(self, vel, rot):
        orientation = self.tip.get_orientation()
        self.sawyer_robot.move_along_velocity_and_orientation(
            tip_velocity=vel,
            orientation=list(np.array(orientation) + rot)
        )


class TargetObject(object):
    def __init__(self, intermediate_waypoints, pixel_waypoint, target_distractor_point, relative_point):
        if len(intermediate_waypoints) == 0:
            raise ValueError("There should be at least one waypoint!")

        # last waypoint is target
        self.intermediate_waypoints = [Dummy(way) for way in intermediate_waypoints]
        self.pixel_waypoint = Dummy(pixel_waypoint)
        self.target_distractor_point = Dummy(target_distractor_point)
        self.relative_point = Dummy(relative_point)

    def get_waypoints(self):
        return self.intermediate_waypoints

    def get_pixel_target(self):
        return self.pixel_waypoint

    def get_final_target(self):
        return self.intermediate_waypoints[-1]

    def get_target_for_distractor_point(self):
        return self.target_distractor_point

    def get_handle(self):
        # returns handle of part of image that will be recorded as pixel
        return self.pixel_waypoint.get_handle()

    def get_position(self, relative_to):
        # target object position relative to camera
        return self.relative_point.get_position(relative_to=relative_to)

    def get_orientation(self, relative_to):
        # target object orientation relative to camera
        return self.relative_point.get_orientation(relative_to=relative_to)


class DiscOffsetGenerator(object):
    def generate_offset(self):
        xy_position_offset = np.random.uniform(-0.15, 0.15, size=2)
        print("XY pos offset", xy_position_offset)
        z_position_offset = np.random.uniform(-0.05, 0.05, size=1)
        xy_orientation_offset = np.random.uniform(-np.pi / 30, np.pi / 30, size=2)
        z_offset = np.random.normal(0, np.pi / 20, size=1)
        res = np.concatenate((xy_position_offset, z_position_offset, xy_orientation_offset, z_offset), axis=0)
        print("Offset", res)
        return res