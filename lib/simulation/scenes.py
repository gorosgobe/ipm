from os.path import abspath, dirname, join

import numpy as np
from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.objects.shape import Shape

from camera_robot import CameraRobot


class Scene(object):
    def __init__(self, scene_file, headless=True, no_distractors=False, random_distractors=False):
        self.scene_file = join(dirname(abspath(__file__)), "../../scenes/", scene_file)
        self.headless = headless
        self.no_distractors = no_distractors
        self.removed_distractors = False
        # TODO: implement
        self.random_distractors = random_distractors
        self.random_distractor_safe_distances = None

    def __enter__(self):
        self.pr = PyRep()
        self.pr.launch(self.scene_file, headless=self.headless)
        self.pr.start()
        return self.pr, self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pr.stop()
        self.pr.shutdown()

    def remove_distractors(self, distractors):
        for dist in distractors:
            dist.remove()
        self.removed_distractors = True

    def get_random_distractor(self):
        primitive_shape = np.random.choice(list(PrimitiveShape))
        if primitive_shape == PrimitiveShape.SPHERE:
            size = [np.random.uniform(0.02, 0.2)] * 3
        else:
            size = list(np.random.uniform(0.02, 0.2, size=3))
        z = 0.55 + (size[2] / 2)
        position = [CameraRobot.get_x_distractor(), CameraRobot.get_y_distractor(), z]
        return Shape.create(
            type=primitive_shape,
            size=size,
            static=True,
            position=position,
            orientation=[0, 0, np.random.uniform(0, 2 * np.pi)],
            color=list(np.random.uniform(0.0, 1.0, size=3))
        ), max(size) + 0.05

    def get_distractors_from_names(self, names):
        if self.no_distractors:
            if not self.removed_distractors:
                self.remove_distractors(Shape(d) for d in names)
            return []
        if self.random_distractors:
            # remove previous shapes
            self.remove_distractors(Shape(d) for d in names)
            shapes = []
            safe_distances = []
            for i in range(len(names)):
                shape, safe_distance = self.get_random_distractor()
                shape.set_name(names[i])
                shapes.append(shape)
                safe_distances.append(safe_distance)
            self.random_distractor_safe_distances = safe_distances
            return shapes
        return [Shape(d) for d in names]

    def get_target(self):
        return Shape("target_cube")

    def get_distractors(self):
        return []

    def get_distractor_safe_distances(self):
        return []

    def get_padding_steps(self):
        raise NotImplementedError

    def get_steps_per_demonstration(self):
        # mean is ~29 for all the generated datasets, without backtracking
        return 29


class SawyerReachCubeScene(Scene):
    SCENE_FILE = "sawyer_reach_cube.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)


class SawyerTextureReachCubeScene(Scene):
    SCENE_FILE = "sawyer_reach_cube_textures.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)


class SawyerTextureDistractorsReachCubeScene(Scene):
    SCENE_FILE = "sawyer_reach_cube_textures_distractors.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)


class CameraTextureReachCubeScene(Scene):
    SCENE_FILE = "camera_reach_cube_textures.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)


class CameraBackgroundObjectsTextureReachCubeScene(Scene):
    SCENE_FILE = "camera_reach_cube_textures_background.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)


class CameraBackgroundObjectsExtendedTextureReachCubeScene(Scene):
    # Same as v1, but camera has 60 degrees for perspective angle and objects are further away
    SCENE_FILE = "camera_reach_cube_textures_background_v2.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)


class CameraBackgroundObjectsTextureReachCubeSceneV3(Scene):
    SCENE_FILE = "camera_reach_cube_textures_background_v3.ttt"
    distractors = [
        "distractor_sphere",
        "distractor_cylinder",
        "distractor_cube"
    ]

    def __init__(self, headless=True, **kwargs):
        super().__init__(self.SCENE_FILE, headless=headless, **kwargs)

    def get_distractors(self):
        return self.get_distractors_from_names(self.distractors)

    def get_distractor_safe_distances(self):
        if self.random_distractors:
            return self.random_distractor_safe_distances
        return [0.1, 0.1, 0.1]


CameraScene1 = CameraBackgroundObjectsTextureReachCubeSceneV3


class CameraScene2(Scene):
    SCENE_FILE = "camera_reach_cube_scene2.ttt"
    distractors = [
        "rectangle_distractor",
        "large_cube_distractor",
        "sphere_distractor"
    ]

    def __init__(self, headless=True, **kwargs):
        super().__init__(self.SCENE_FILE, headless=headless, **kwargs)

    def get_distractors(self):
        return self.get_distractors_from_names(self.distractors)

    def get_distractor_safe_distances(self):
        if self.random_distractors:
            return self.random_distractor_safe_distances
        return [0.15, 0.15, 0.1]


class CameraScene3(Scene):
    SCENE_FILE = "camera_reach_cube_scene3.ttt"
    distractors = [
        "distractor_cylinder",
        *[f"distractor_sphere{i}" for i in range(1, 11)]
    ]

    def __init__(self, headless=True, **kwargs):
        super().__init__(self.SCENE_FILE, headless=headless, **kwargs)

    def get_distractors(self):
        return self.get_distractors_from_names(self.distractors)

    def get_distractor_safe_distances(self):
        if self.random_distractors:
            return self.random_distractor_safe_distances
        return [0.10, *[0.05 for i in range(1, 11)]]


class CameraScene4(Scene):
    SCENE_FILE = "camera_reach_cube_scene4.ttt"
    distractors = [
        "distractor_cylinder",
        "distractor_cube1",
        "distractor_cube2_long",
        "distractor_cube3",
        "distractor_cube4",
        "distractor_cube5",
        "distractor_cube6"
    ]

    def __init__(self, headless=True, **kwargs):
        super().__init__(self.SCENE_FILE, headless=headless, **kwargs)

    def get_distractors(self):
        if self.no_distractors or self.random_distractors:
            return self.get_distractors_from_names(self.distractors + ["distractor_cylinder_sphere"])
        return self.get_distractors_from_names(self.distractors)

    def get_distractor_safe_distances(self):
        if self.random_distractors:
            return self.random_distractor_safe_distances
        return [0.075, 0.10, 0.05, 0.05, 0.1, 0.1, 0.1]


class CameraScene5(Scene):
    SCENE_FILE = "camera_reach_cube_scene5.ttt"
    distractors = [
        "distractor_cube",
        "distractor_cube1",
        "distractor_cube2",
        "distractor_cube3",
        "distractor_cube4",
        "distractor_disc",
        "distractor_plane",
        "distractor_sphere"
    ]

    def __init__(self, headless=True, **kwargs):
        super().__init__(self.SCENE_FILE, headless=headless, **kwargs)

    def get_distractors(self):
        return self.get_distractors_from_names(self.distractors)

    def get_distractor_safe_distances(self):
        if self.random_distractors:
            return self.random_distractor_safe_distances
        return [0.1, 0.1, 0.1, 0.1, 0.1, 0.075, 0.05, 0.1]


class CameraSceneSimplest(Scene):
    SCENE_FILE = "camera_reach_cube_simplest.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)
