from os.path import abspath, dirname, join

from pyrep import PyRep


class Scene(object):
    def __init__(self, scene_file, headless=True):
        self.scene_file = scene_file
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
    SCENE_FILE = join(dirname(abspath(__file__)), "scenes/sawyer_reach_cube.ttt")

    def __init__(self, headless=True):
        super(SawyerReachCubeScene, self).__init__(SawyerReachCubeScene.SCENE_FILE, headless=headless)


class SawyerTextureReachCubeScene(Scene):
    SCENE_FILE = join(dirname(abspath(__file__)), "scenes/sawyer_reach_cube_textures.ttt")

    def __init__(self, headless=True):
        super(SawyerTextureReachCubeScene, self).__init__(SawyerTextureReachCubeScene.SCENE_FILE, headless=headless)


class SawyerTextureDistractorsReachCubeScene(Scene):
    SCENE_FILE = join(dirname(abspath(__file__)), "scenes/sawyer_reach_cube_textures_distractors.ttt")

    def __init__(self, headless=True):
        super(SawyerTextureDistractorsReachCubeScene, self).__init__(SawyerTextureDistractorsReachCubeScene.SCENE_FILE,
                                                                     headless=headless)
