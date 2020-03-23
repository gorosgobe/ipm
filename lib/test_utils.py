import enum

from lib.controller import IdentityCropper, ControllerType, TruePixelROI, CropDeviationSampler
from lib.scenes import CameraScene5, CameraScene1, CameraScene4, CameraScene3, CameraScene2


def get_scene_and_test_scene_configuration(model_name):
    if "scene2" in model_name:
        s = CameraScene2
        test = "test_demonstrations/scene2_test.json"
    elif "scene3" in model_name:
        s = CameraScene3
        test = "test_demonstrations/scene3_test.json"
    elif "scene4" in model_name:
        s = CameraScene4
        test = "test_demonstrations/scene4_test.json"
    elif "scene5" in model_name:
        s = CameraScene5
        test = "test_demonstrations/scene5_test.json"
    elif "rand" in model_name.lower() or "scene1" in model_name:
        s = CameraScene1
        test = "test_demonstrations/scene1_test.json"
    else:
        raise ValueError("Unknown scene and test scene configuration")

    return s, test


class TestConfig(enum.Enum):
    BASELINE = 0,
    FULL_IMAGE = 1,
    ATTENTION_64 = 2,
    ATTENTION_32 = 3,
    ATTENTION_COORD_64 = 4,
    ATTENTION_COORD_32 = 5,
    COORD_32_ST_100 = 6


def get_testing_configs(camera_robot, target_cube):
    return {
        TestConfig.BASELINE:
            {
                "cropper": IdentityCropper(),
                "c_type": ControllerType.RELATIVE_POSITION_AND_ORIENTATION
            },
        TestConfig.FULL_IMAGE:
            {
                "cropper": IdentityCropper(),
                "c_type": ControllerType.DEFAULT
            },
        TestConfig.ATTENTION_64:
            {
                "cropper": TruePixelROI(480 // 2, 640 // 2, camera_robot.get_movable_camera(), target_cube),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
        TestConfig.ATTENTION_32:
            {
                "cropper": TruePixelROI(480 // 4, 640 // 4, camera_robot.get_movable_camera(), target_cube),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
        TestConfig.ATTENTION_COORD_64:
            {
                "cropper": TruePixelROI(480 // 2, 640 // 2, camera_robot.get_movable_camera(), target_cube,
                                        add_spatial_maps=True),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
        TestConfig.ATTENTION_COORD_32:
            {
                "cropper": TruePixelROI(480 // 4, 640 // 4, camera_robot.get_movable_camera(), target_cube,
                                        add_spatial_maps=True),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
        TestConfig.COORD_32_ST_100:
            {
                "cropper": TruePixelROI(480 // 4, 640 // 4, camera_robot.get_movable_camera(), target_cube,
                                        add_spatial_maps=True, crop_deviation_sampler=CropDeviationSampler(100)),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
    }