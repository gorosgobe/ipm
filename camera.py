import math

import cv2
import numpy as np
from pyrep.backend import sim
from pyrep.objects.cartesian_path import CartesianPath


class Camera(object):
    def __init__(self, name):
        self.name = name
        self.vision_sensor_handle = sim.simGetObjectHandle(self.name)
        self.resolution = sim.simGetVisionSensorResolution(self.vision_sensor_handle)

    def get_image(self):
        return sim.simGetVisionSensorImage(self.vision_sensor_handle, self.resolution)

    def save_current_image(self, path):
        return Camera.save_image(path, self.get_image())

    @staticmethod
    def save_image(path, image):
        image = cv2.convertScaleAbs(image, alpha=(255.0))
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def get_position(self):
        return sim.simGetObjectPosition(self.vision_sensor_handle, -1)


class WristCamera(Camera):
    VISION_SENSOR = "Sawyer_wristCamera"

    def __init__(self):
        super(WristCamera, self).__init__(WristCamera.VISION_SENSOR)


class MovableCamera(Camera):
    VISION_SENSOR = "movable_camera"

    def __init__(self, show_paths=False):
        super(MovableCamera, self).__init__(MovableCamera.VISION_SENSOR)
        self.show_paths = show_paths
        self.initial_position = sim.simGetObjectPosition(self.vision_sensor_handle, -1)
        self.initial_orientation = sim.simGetObjectOrientation(self.vision_sensor_handle, -1)

    def _set_offset_position(self, offset, current_position):
        # Expects np array for mathematical operations, passed as list to API
        sim.simSetObjectPosition(self.vision_sensor_handle, -1, list(current_position + offset))

    def set_initial_offset_position(self, offset):
        self._set_offset_position(offset, np.array(self.initial_position))
        sim.simSetObjectOrientation(self.vision_sensor_handle, -1, self.initial_orientation)

    def set_orientation(self, point_towards):
        path = CartesianPath.create(show_line=self.show_paths, show_orientation=self.show_paths, automatic_orientation=True)
        current_position = sim.simGetObjectPosition(self.vision_sensor_handle, -1)
        for i in range(3):
            current_position.append(0.0)

        point_towards = list(point_towards)
        for i in range(3):
            point_towards.append(0.0)

        path.insert_control_points([current_position, point_towards])
        # relative orientation, at the end of the path
        _, pose = path.get_pose_on_path(1)
        sim.simSetObjectOrientation(self.vision_sensor_handle, -1, pose)

    def move_along_velocity(self, velocity):
        step = sim.simGetSimulationTimeStep()
        position = sim.simGetObjectPosition(self.vision_sensor_handle, -1)
        sim.simSetObjectPosition(self.vision_sensor_handle, -1, list(position + step * velocity))
