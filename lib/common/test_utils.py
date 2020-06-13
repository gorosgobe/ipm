import enum
import json
import math
import os
from collections import OrderedDict

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from lib.cv.controller import IdentityCropper, ControllerType, TruePixelROI, CropDeviationSampler


def get_scene_and_test_scene_configuration(model_name):
    from lib.simulation.scenes import CameraScene5, CameraScene1, CameraScene4, CameraScene3, CameraScene2, \
        SawyerInsertDiscScene
    if "scene2" in model_name or "randist2" in model_name or "nodist2" in model_name:
        s = CameraScene2
        test = "test_demonstrations/scene2_test.json"
        dist_config = "test_demonstrations/random2_test.json"
    elif "scene3" in model_name or "randist3" in model_name or "nodist3" in model_name:
        s = CameraScene3
        test = "test_demonstrations/scene3_test.json"
        dist_config = "test_demonstrations/random3_test.json"
    elif "scene4" in model_name or "randist4" in model_name or "nodist4" in model_name:
        s = CameraScene4
        test = "test_demonstrations/scene4_test.json"
        dist_config = "test_demonstrations/random4_test.json"
    elif "scene5" in model_name or "randist5" in model_name or "nodist5" in model_name:
        s = CameraScene5
        test = "test_demonstrations/scene5_test.json"
        dist_config = "test_demonstrations/random5_test.json"
    elif "scene1" in model_name or "randist1" in model_name or "nodist1" in model_name:
        s = CameraScene1
        test = "test_demonstrations/scene1_test.json"
        dist_config = "test_demonstrations/random1_test.json"
    elif "disc" in model_name:
        s = SawyerInsertDiscScene
        test = ...
        dist_config = "test_demonstrations/disc_test.json"
    else:
        raise ValueError("Unknown scene and test scene configuration")

    return s, test, dist_config


class TestConfig(enum.Enum):
    BASELINE = 0
    FULL_IMAGE = 1
    ATTENTION_64 = 2
    ATTENTION_32 = 3
    ATTENTION_COORD_64 = 4
    ATTENTION_COORD_32 = 5
    COORD_32_ST_100 = 6
    ATTENTION_COORD_ROT_32 = 7
    DSAE = 8
    RECURRENT_FULL = 9
    DSAE_CHOOSE = 10
    STN = 11
    RECURRENT_ATTENTION_COORD_32 = 12
    RECURRENT_BASELINE = 13


def get_testing_configs(camera_robot, pixel_target):
    return {
        TestConfig.BASELINE:
            {
                "cropper": IdentityCropper(),
                "c_type": ControllerType.RELATIVE_POSITION_AND_ORIENTATION
            },
        TestConfig.RECURRENT_BASELINE:
            {
                "cropper": IdentityCropper(),
                "c_type": ControllerType.RELATIVE_POSITION_AND_ORIENTATION
            },
        TestConfig.RECURRENT_FULL:
            {
                "cropper": IdentityCropper(),
                "c_type": ControllerType.DEFAULT
            },
        TestConfig.STN:
            {
                "cropper": TruePixelROI(480, 640, camera_robot.get_movable_camera(), pixel_target,
                                        add_spatial_maps=True),
                "c_type": ControllerType.DEFAULT
            },
        TestConfig.DSAE_CHOOSE:
            {
                "cropper": IdentityCropper(),  # will be overriden
                "c_type": ControllerType.DEFAULT
            },
        TestConfig.FULL_IMAGE:
            {
                "cropper": IdentityCropper(),
                "c_type": ControllerType.DEFAULT
            },
        TestConfig.ATTENTION_64:
            {
                "cropper": TruePixelROI(480 // 2, 640 // 2, camera_robot.get_movable_camera(), pixel_target),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
        TestConfig.ATTENTION_32:
            {
                "cropper": TruePixelROI(480 // 4, 640 // 4, camera_robot.get_movable_camera(), pixel_target),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
        TestConfig.ATTENTION_COORD_64:
            {
                "cropper": TruePixelROI(480 // 2, 640 // 2, camera_robot.get_movable_camera(), pixel_target,
                                        add_spatial_maps=True),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
        TestConfig.ATTENTION_COORD_32:
            {
                "cropper": TruePixelROI(480 // 4, 640 // 4, camera_robot.get_movable_camera(), pixel_target,
                                        add_spatial_maps=True),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
        TestConfig.RECURRENT_ATTENTION_COORD_32:
            {
                "cropper": TruePixelROI(480 // 4, 640 // 4, camera_robot.get_movable_camera(), pixel_target,
                                        add_spatial_maps=True),
                "c_type": ControllerType.DEFAULT
            },
        TestConfig.ATTENTION_COORD_ROT_32:
            {
                "cropper": TruePixelROI(480 // 4, 640 // 4, camera_robot.get_movable_camera(), pixel_target,
                                        add_spatial_maps=True, add_r_map=True),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
        TestConfig.COORD_32_ST_100:
            {
                "cropper": TruePixelROI(480 // 4, 640 // 4, camera_robot.get_movable_camera(), pixel_target,
                                        add_spatial_maps=True, crop_deviation_sampler=CropDeviationSampler(100)),
                "c_type": ControllerType.TOP_LEFT_BOTTOM_RIGHT_PIXELS
            },
        TestConfig.DSAE:
            {
                "cropper": IdentityCropper(),
                "c_type": ControllerType.DEFAULT
            }
    }


def get_latex(means, stds, display_all_values=True, display_std_means=False):
    # if display_all_values is False, then we display mean +- std
    def round_to_two_decimals_list(li):
        return list(map(lambda x: round(x * 100, 2), li))

    def _round(x):
        return round(x * 100, 2)

    # means is dictionary
    res_latex = ""
    count = 0
    for name in means:
        list_mean = means[name]
        list_std = stds[name]
        count += 1
        if display_all_values:
            res_latex += f"{round_to_two_decimals_list(list_mean)}/{round_to_two_decimals_list(list_std)} & "
        else:
            res_latex += f"{_round(np.mean(np.array(list_mean)))}$\pm${_round(np.std(np.array(list_mean)))}"
            if display_std_means:
                res_latex += f"{_round(np.mean(np.array(list_std)))}$\pm${_round(np.std(np.array(list_std)))}"
            res_latex += " & "
        if count % 6 == 0:
            res_latex += "\n"
            count = 0

    return res_latex


def load_test(test_name):
    achieved = None
    with open(test_name, "r") as f:
        content = json.loads(f.read())
        minimum_distances = content["min_distances"]
        fixed_step_distances = content["fixed_steps_distances"]
        if "achieved" in content:
            achieved = content["achieved"]

    return minimum_distances, fixed_step_distances, achieved


def compute_95_interval(values):
    avg = np.mean(np.array(values))
    n = len(values)
    variance_estimate = (1 / (n - 1)) * sum(list(map(lambda v: (v - avg) ** 2, values)))
    alpha = 0.05
    t = stats.t.ppf(1 - (alpha / 2), n - 1)
    h = (t * math.sqrt(variance_estimate)) / math.sqrt(n)
    return avg, h


def get_achieved_and_target(distances, minimum_distances, special_distance):
    test_achieved = []
    special_distance_count = len(list(filter(lambda x: x <= special_distance, minimum_distances.values())))
    for target_distance in distances:
        achieved = len(list(filter(lambda x: x <= target_distance, minimum_distances.values())))
        test_achieved.append((target_distance, achieved))

    return test_achieved, special_distance_count


def plot_achieved(achieved_plot, legend_names=None):
    mean_intervals = OrderedDict()
    for t in achieved_plot:
        name, achieved_data = t
        t_base = os.path.basename(name)
        if t_base not in mean_intervals:
            mean_intervals[t_base] = [achieved_data]
        else:
            mean_intervals[t_base].append(achieved_data)

    for base in mean_intervals:
        # compute mean and std at every datapoint
        lines = mean_intervals[base]
        num_achieved = []
        target_distances = []
        for line in lines:
            target_distances, line_achieved = zip(*line)
            num_achieved.append(line_achieved)
        mean_achieved = np.mean(num_achieved, axis=0)
        std_achieved = np.std(num_achieved, axis=0)
        plt.plot(target_distances, mean_achieved, label=base if legend_names is None else legend_names[base])
        plt.fill_between(target_distances, mean_achieved - std_achieved, mean_achieved + std_achieved, alpha=0.25)

    plt.legend()
    plt.xlabel("Threshold distance (m)")
    plt.ylabel("Success rate")
    plt.show()


def add_value_to_test(collection, test_name, res):
    if test_name not in collection:
        collection[test_name] = [res]
    else:
        collection[test_name].append(res)


def draw_crop(image, tl_gt, br_gt, red=False, size=1):
    color = (0, 255, 0) if not red else (255, 0, 0)
    cv2.rectangle(image, tuple(tl_gt), tuple(br_gt), color, size)


def downsample_coordinates(x, y, og_width, og_height, to_width, to_height):
    downsampled_x = int((x / og_width) * to_width)
    downsampled_y = int((y / og_height) * to_height)
    return downsampled_x, downsampled_y


def get_distance_between_boxes(tl_gt, predicted_tl, *_args):
    return np.linalg.norm(np.array(tl_gt) - np.array(predicted_tl))


def calculate_IoU(tl_gt, br_gt, predicted_tl, predicted_gt):
    # TODO: ?
    pass


def get_distances_between_chosen_features_and_pixels(features, pixels, width=128, height=96):
    assert pixels.shape == (len(features), 2)
    # features (length episode, k * 2)
    # pixels (length episode, 2)
    normalised_features = (features + 1) / 2
    bounds = np.array([width - 1, height - 1], dtype=np.float32).reshape((1, 1, 2))
    normalised_features = normalised_features.reshape((features.shape[0], -1, 2)) * bounds
    difference = normalised_features - pixels.reshape((pixels.shape[0], 1, 2))
    # norms is of size (length episode, k)
    norms = np.linalg.norm(difference, axis=-1)
    return norms
