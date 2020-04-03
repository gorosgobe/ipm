from os.path import abspath, dirname, join

from pyrep import PyRep
from pyrep.objects.shape import Shape


class Scene(object):
    def __init__(self, scene_file, headless=True):
        self.scene_file = join(dirname(abspath(__file__)), "../../scenes/", scene_file)
        self.headless = headless

    def __enter__(self):
        self.pr = PyRep()
        self.pr.launch(self.scene_file, headless=self.headless)
        self.pr.start()
        return self.pr, self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pr.stop()
        self.pr.shutdown()

    def get_target(self):
        return Shape("target_cube")

    def get_distractors(self):
        return []

    def get_distractor_safe_distances(self):
        return []

    def get_padding_steps(self):
        raise NotImplementedError

    def get_steps_per_demonstration(self):
        return NotImplementedError


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

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)

    def get_distractors(self):
        return [Shape("distractor_sphere"), Shape("distractor_cylinder"), Shape("distractor_cube")]

    def get_distractor_safe_distances(self):
        return [0.1, 0.1, 0.1]

    def get_steps_per_demonstration(self):
        # mean is 29.1, without backtracking
        return 29


CameraScene1 = CameraBackgroundObjectsTextureReachCubeSceneV3


class CameraScene2(Scene):
    SCENE_FILE = "camera_reach_cube_scene2.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)

    def get_distractors(self):
        return [Shape("rectangle_distractor"), Shape("large_cube_distractor"), Shape("sphere_distractor")]

    def get_distractor_safe_distances(self):
        return [0.15, 0.15, 0.1]


class CameraScene3(Scene):
    SCENE_FILE = "camera_reach_cube_scene3.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)

    def get_distractors(self):
        return [Shape("distractor_cylinder"), *[Shape(f"distractor_sphere{i}") for i in range(1, 11)]]

    def get_distractor_safe_distances(self):
        return [0.10, *[0.05 for i in range(1, 11)]]


class CameraScene4(Scene):
    SCENE_FILE = "camera_reach_cube_scene4.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)

    def get_distractors(self):
        return [
            Shape("distractor_cylinder"),
            Shape("distractor_cube1"),
            Shape("distractor_cube2_long"),
            Shape("distractor_cube3"),
            Shape("distractor_cube4"),
            Shape("distractor_cube5"),
            Shape("distractor_cube6")
        ]

    def get_distractor_safe_distances(self):
        return [0.075, 0.10, 0.05, 0.05, 0.1, 0.1, 0.1]


class CameraScene5(Scene):
    SCENE_FILE = "camera_reach_cube_scene5.ttt"

    def __init__(self, headless=True):
        super().__init__(self.SCENE_FILE, headless=headless)

    def get_distractors(self):
        return [
            Shape("distractor_cube"),
            Shape("distractor_cube1"),
            Shape("distractor_cube2"),
            Shape("distractor_cube3"),
            Shape("distractor_cube4"),
            Shape("distractor_disc"),
            Shape("distractor_plane"),
            Shape("distractor_sphere")
        ]

    def get_distractor_safe_distances(self):
        return [0.1, 0.1, 0.1, 0.1, 0.1, 0.075, 0.05, 0.1]
