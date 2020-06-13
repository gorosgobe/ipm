import json
from itertools import cycle
from os.path import abspath, dirname, join

import numpy as np
from pyrep import PyRep
from pyrep.objects.shape import Shape

from lib.simulation.distractor import Distractor
from lib.simulation.sawyer_camera_adapter import TargetObject
from lib.simulation.sim_gt_estimators import SimGTVelocityEstimator, SimGTOrientationEstimator, ContinuousGTEstimator


class Scene(object):
    def __init__(self, scene_file, headless=True, no_distractors=False, random_distractors=False,
                 distractors_from=None):
        self.scene_file = join(dirname(abspath(__file__)), "../../scenes/", scene_file)
        self.headless = headless
        self.no_distractors = no_distractors
        self.removed_distractors = False
        self.random_distractors = random_distractors
        self.random_distractor_safe_distances = None
        self.distractors_from = distractors_from
        self.test_distractors = None
        self.target = None

    def __enter__(self):
        # load test random distractors if available
        if self.distractors_from is not None:
            with open(self.distractors_from, "r") as f:
                content = f.read()
            self.test_distractors = json.loads(content)["test_distractors"]
            self.test_distractor_iterator = iter(self.get_test_distractor())

        self.pr = PyRep()
        self.pr.launch(self.scene_file, headless=self.headless)
        self.pr.start()
        self.target = self._get_target()
        return self.pr, self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pr.stop()
        self.pr.shutdown()

    def remove_distractors(self, distractors):
        for dist in distractors:
            dist.remove()
        self.removed_distractors = True

    def get_test_distractor(self):
        for count in cycle(range(100)):
            for single_dist_data in self.test_distractors[count]:
                distractor = Distractor(**single_dist_data)
                yield distractor.get_shape(), distractor.get_safe_distance()

    @staticmethod
    def get_random_distractor():
        random_distractor = Distractor()
        return random_distractor.get_shape(), random_distractor.get_safe_distance()

    def get_distractors_from_names(self, names):
        if self.no_distractors:
            if not self.removed_distractors:
                self.remove_distractors(Shape(d) for d in names)
            return []
        if self.random_distractors:
            try:
                # remove previous shapes
                self.remove_distractors(Shape(d) for d in names)
            except:
                pass
            shapes = []
            safe_distances = []
            for i in range(len(names)):
                if self.test_distractors is None:
                    shape, safe_distance = self.get_random_distractor()
                else:
                    shape, safe_distance = next(self.test_distractor_iterator)
                shape.set_name(names[i])
                shapes.append(shape)
                safe_distances.append(safe_distance)
            self.random_distractor_safe_distances = safe_distances
            return shapes
        return [Shape(d) for d in names]

    def _get_target(self):
        return TargetObject(
            intermediate_waypoints=["target_above_cube"],
            pixel_waypoint="target_cube_dummy",
            target_distractor_point="target_cube_dummy",
            relative_point="target_cube_dummy"
        )

    def is_break_early(self):
        return False

    def get_target(self):
        return self.target

    def get_distractors(self):
        return []

    def get_distractor_safe_distances(self):
        return []

    def get_padding_steps(self):
        raise NotImplementedError

    def get_steps_per_demonstration(self):
        # mean is ~29 for all the generated datasets, without backtracking
        return 29

    def get_sim_gt_velocity(self, generating=False, discontinuity=True):
        target_above_cube = self.get_target().get_final_target().get_position()
        return SimGTVelocityEstimator(target_above_cube, generating=generating, discontinuity=discontinuity)

    def get_sim_gt_orientation(self):
        target_above_cube = self.get_target().get_final_target().get_position()
        return SimGTOrientationEstimator(target_above_cube, [-np.pi, 0, -np.pi / 2])


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


class SawyerInsertDiscScene(Scene):
    SCENE_FILE = "sawyer_insert_disc.ttt"
    distractors = ["dist1", "dist2", "dist3", "dist4"]

    def __init__(self, headless=True, **kwargs):
        super().__init__(self.SCENE_FILE, headless=headless, **kwargs)
        self.gt_estimator = None

    def is_break_early(self):
        return True

    def _get_target(self):
        return TargetObject(
            intermediate_waypoints=["peg_waypoint_reach", "peg_waypoint_end"],
            pixel_waypoint="peg_waypoint_end",
            target_distractor_point="peg_target",
            relative_point="peg_waypoint_reach"
        )

    def get_sim_gt_velocity(self, generating=False, discontinuity=True):
        self.gt_estimator = ContinuousGTEstimator(
            self.target.get_waypoints(),
            generating=generating,
            reduce_factors=[1.25, 2]
        )
        return self.gt_estimator

    def get_sim_gt_orientation(self):
        if self.gt_estimator is None:
            raise ValueError("Call get sim GT velocity first")
        return self.gt_estimator

    def get_distractors(self):
        return self.get_distractors_from_names(self.distractors)

    def get_distractor_safe_distances(self):
        if self.random_distractors:
            return self.random_distractor_safe_distances
        return [0.05] * 4

    def get_steps_per_demonstration(self):
        return 33

