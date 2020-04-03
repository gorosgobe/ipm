import math

import numpy as np
from pyrep.backend import sim
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.objects.vision_sensor import VisionSensor

from common import utils


class Camera(object):
    def __init__(self, name):
        self.name = name
        self.vision_sensor = VisionSensor(self.name)
        self.resolution = self.vision_sensor.get_resolution()
        self.perspective_angle = sim.simGetObjectFloatParameter(
            self.vision_sensor.get_handle(), sim.sim_visionfloatparam_perspective_angle
        )

    def get_image(self):
        image = self.vision_sensor.capture_rgb()
        assert 0 <= image[0][0][0] <= 1
        return image

    def save_current_image(self, path):
        return utils.save_image(path, self.get_image())

    def get_position(self, relative_to=None):
        return self.vision_sensor.get_position(relative_to=relative_to)

    def set_position(self, position, relative_to=None):
        if not isinstance(position, list):
            position = list(position)
        return self.vision_sensor.set_position(position, relative_to=relative_to)

    def get_orientation(self, relative_to=None):
        return self.vision_sensor.get_orientation(relative_to=relative_to)

    def set_orientation(self, orientation, relative_to=None):
        if not isinstance(orientation, list):
            orientation = list(orientation)
        return self.vision_sensor.set_orientation(orientation, relative_to=relative_to)

    def get_handle(self):
        return self.vision_sensor.get_handle()


class WristCamera(Camera):
    VISION_SENSOR = "Sawyer_wristCamera"

    def __init__(self):
        super(WristCamera, self).__init__(WristCamera.VISION_SENSOR)


class MovableCamera(Camera):
    VISION_SENSOR = "movable_camera"

    def __init__(self, show_paths=False, initial_position=None):
        super(MovableCamera, self).__init__(MovableCamera.VISION_SENSOR)
        self.show_paths = show_paths
        if initial_position is not None:
            self.set_position(initial_position)
        self.initial_position = self.get_position()
        self.initial_orientation = self.get_orientation()
        self.ratio = self.resolution[0] / self.resolution[1]
        # from coppelia sim forum
        if self.ratio > 1:
            self.angle_x = self.perspective_angle
            self.angle_y = 2 * math.atan(math.tan(self.perspective_angle / 2) / self.ratio)
        else:
            self.angle_x = 2 * math.atan(math.tan(self.perspective_angle / 2) * self.ratio)
            self.angle_y = self.perspective_angle

    def _set_offset_position(self, offset, current_position):
        # Expects np array for mathematical operations, passed as list to API
        self.set_position(list(current_position + offset))

    def set_initial_offset_position(self, offset):
        self._set_offset_position(offset, np.array(self.initial_position))
        self.set_orientation(self.initial_orientation)

    def set_orientation_towards(self, point_towards):
        path = CartesianPath.create(show_line=self.show_paths, show_orientation=self.show_paths,
                                    automatic_orientation=True)
        current_position = self.get_position()
        for i in range(3):
            current_position.append(0.0)

        point_towards = list(point_towards)
        for i in range(3):
            point_towards.append(0.0)

        path.insert_control_points([current_position, point_towards])
        # relative orientation, at the end of the path
        _, pose = path.get_pose_on_path(1)
        self.set_orientation(pose, path.get_handle())

    def move_along_velocity(self, velocity):
        step = sim.simGetSimulationTimeStep()
        position = self.get_position()
        self.set_position(list(position + step * velocity))

    def _compute_canvas_position(self, handle=None, point=None):
        # point, if supplied, has to be in camera space
        if handle is None and point is None:
            raise Exception("Both handle and point should not be None")
        if handle is not None:
            point = sim.simGetObjectPosition(handle, self.get_handle())
        x = point[0]
        y = point[1]
        z = point[2]
        # ignore near plane, set to z = 1
        return np.array([-x / z, y / z, 1])

    def _compute_pixel_position_from_canvas(self, canvas_position):
        half_height_field_of_view = self.angle_y * 0.5
        h = math.tan(half_height_field_of_view)

        half_width_field_view = self.angle_x * 0.5
        w = math.tan(half_width_field_view)

        width = 2 * w
        height = 2 * h
        normalised_x = (canvas_position[0] + width / 2) / width
        normalised_y = 1 - (canvas_position[1] + height / 2) / height
        raster_x = int(normalised_x * self.resolution[0])
        raster_y = int(normalised_y * self.resolution[1])
        return raster_x, raster_y

    def compute_pixel_position(self, handle, debug=False):
        canvas_position = self._compute_canvas_position(handle)
        # for debugging, aim is to see the axis
        points_to_draw = []
        if debug:
            towards_x_axis = map(
                self._compute_pixel_position_from_canvas,
                [self._compute_canvas_position(point=[x, 0.0, 1.0]) for x in np.arange(0.0, 1.1, 0.1)]
            )
            towards_y_axis = map(
                self._compute_pixel_position_from_canvas,
                [self._compute_canvas_position(point=[0.0, y, 1.0]) for y in np.arange(0.0, 1.1, 0.1)]
            )
            towards_z_axis = map(
                self._compute_pixel_position_from_canvas,
                [self._compute_canvas_position(point=[0.0, 0.0, z + 0.1]) for z in np.arange(0.0, 1.1, 0.1)]
            )
            points_to_draw.append(towards_x_axis)
            points_to_draw.append(towards_y_axis)
            points_to_draw.append(towards_z_axis)
        return self._compute_pixel_position_from_canvas(canvas_position), points_to_draw

    def add_to_orientation(self, rotation):
        self.set_orientation(np.array(self.get_orientation()) + rotation)
