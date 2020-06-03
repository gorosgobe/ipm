import numpy as np
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape

from lib.simulation.camera_robot import CameraRobot


class Distractor(object):
    def __init__(self, type=None, size=None, static=None, position=None, orientation=None, color=None):
        if type is not None:
            primitive_shape = self.get_prim_shape_from_idx(type)
        else:
            primitive_shape = np.random.choice(list(PrimitiveShape))

        self.primitive_shape = primitive_shape

        if primitive_shape == PrimitiveShape.SPHERE:
            self.size = size or [np.random.uniform(0.02, 0.2)] * 3
        else:
            self.size = size or list(np.random.uniform(0.02, 0.2, size=3))

        z = 0.55 + (self.size[2] / 2)
        self.position = position or [CameraRobot.get_x_distractor(), CameraRobot.get_y_distractor(), z]
        self.orientation = orientation or [0, 0, np.random.uniform(0, 2 * np.pi)]
        self.color = color or list(np.random.uniform(0.0, 1.0, size=3))
        self.static = static or True

        self.shape = Shape.create(
            type=self.primitive_shape,
            size=self.size,
            static=self.static,
            position=self.position,
            orientation=self.orientation,
            color=self.color
        )

    def get_safe_distance(self):
        return max(self.size) + 0.05

    @staticmethod
    def get_prim_shape_from_idx(type):
        if type == 0:
            primitive_shape = PrimitiveShape.CUBOID
        elif type == 1:
            primitive_shape = PrimitiveShape.SPHERE
        elif type == 2:
            primitive_shape = PrimitiveShape.CYLINDER
        else:
            primitive_shape = PrimitiveShape.CONE
        return primitive_shape

    @staticmethod
    def get_idx_from_prim_shape(type):
        if type == PrimitiveShape.CUBOID:
            idx = 0
        elif type == PrimitiveShape.SPHERE:
            idx = 1
        elif type == PrimitiveShape.CYLINDER:
            idx = 2
        else:
            idx = 3
        return idx

    def serialise(self):
        return dict(
            type=self.get_idx_from_prim_shape(self.primitive_shape),
            size=self.size,
            static=self.static,
            position=self.position,
            orientation=self.orientation,
            color=self.color
        )

    def get_position(self):
        return self.position

    def set_position(self, position):
        self.position = position

    def get_size(self):
        return self.size

    def get_shape(self):
        return self.shape
