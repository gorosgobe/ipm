from os.path import abspath, dirname, join

from pyrep import PyRep


class Scene(object):
    def __init__(self, scene_file, headless=True):
        self.scene_file = join(dirname(abspath(__file__)), "../scenes/", scene_file)
        self.headless = headless

    def __enter__(self):
        self.pr = PyRep()
        self.pr.launch(self.scene_file, headless=self.headless)
        self.pr.start()
        return self.pr

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pr.stop()
        self.pr.shutdown()


class SawyerReachCubeScene(Scene):
    SCENE_FILE = "sawyer_reach_cube.ttt"

    def __init__(self, headless=True):
        super(SawyerReachCubeScene, self).__init__(self.SCENE_FILE, headless=headless)


class SawyerTextureReachCubeScene(Scene):
    SCENE_FILE = "sawyer_reach_cube_textures.ttt"

    def __init__(self, headless=True):
        super(SawyerTextureReachCubeScene, self).__init__(self.SCENE_FILE, headless=headless)


class SawyerTextureDistractorsReachCubeScene(Scene):
    SCENE_FILE = "sawyer_reach_cube_textures_distractors.ttt"

    def __init__(self, headless=True):
        super(SawyerTextureDistractorsReachCubeScene, self).__init__(self.SCENE_FILE,
                                                                     headless=headless)


class CameraTextureReachCubeScene(Scene):
    SCENE_FILE = "camera_reach_cube_textures.ttt"

    def __init__(self, headless=True):
        super(CameraTextureReachCubeScene, self).__init__(self.SCENE_FILE, headless=headless)


class CameraBackgroundObjectsTextureReachCubeScene(Scene):
    SCENE_FILE = "camera_reach_cube_textures_background.ttt"

    def __init__(self, headless=True):
        super(CameraBackgroundObjectsTextureReachCubeScene, self).__init__(self.SCENE_FILE,
                                                                           headless=headless)


class CameraBackgroundObjectsExtendedTextureReachCubeScene(Scene):
    # Same as v1, but camera has 60 degrees for perspective angle and objects are further away
    SCENE_FILE = "camera_reach_cube_textures_background_v2.ttt"

    def __init__(self, headless=True):
        super(CameraBackgroundObjectsExtendedTextureReachCubeScene, self).__init__(self.SCENE_FILE,
                                                                                   headless=headless)
