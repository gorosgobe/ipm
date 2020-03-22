import csv
import json
import os

import cv2
import torch
import torchvision


class ResizeTransform(object):
    def __init__(self, size):
        self.size = size  # tuple, like (128, 96)

    def __call__(self, image):
        return cv2.resize(image, dsize=self.size)


def set_up_cuda(seed, set_up_seed=True):
    # set up GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda and set_up_seed:
        torch.cuda.manual_seed(seed)
    print("Using GPU: {}".format(use_cuda))

    return device


def get_preprocessing_transforms(size, is_coord=False):
    normalisation = torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    if is_coord:
        # to account for spatial maps
        normalisation = torchvision.transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5])

    transforms_without_resize = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalisation,
    ])

    preprocessing_transforms_complete = torchvision.transforms.Compose([ResizeTransform(size),
                                                                        transforms_without_resize
                                                                        ])
    return preprocessing_transforms_complete, transforms_without_resize


def get_demonstrations(dataset, split, limit_train_coeff=-1):
    total_demonstrations = dataset.get_num_demonstrations()

    training_demonstrations, n_training_dems = dataset.get_split(split[0], total_demonstrations, 0)
    val_demonstrations, n_val_dems = dataset.get_split(split[1], total_demonstrations, n_training_dems)
    test_demonstrations, n_test_dems = dataset.get_split(split[2], total_demonstrations, n_training_dems + n_val_dems)

    # Limited dataset
    if limit_train_coeff != -1:
        training_demonstrations, n_training_dems = dataset.get_split(limit_train_coeff, total_demonstrations, 0)

    print("Training demonstrations: ", n_training_dems, len(training_demonstrations))
    print("Validation demonstrations: ", n_val_dems, len(val_demonstrations))
    print("Test demonstrations: ", n_test_dems, len(test_demonstrations))

    return training_demonstrations, val_demonstrations, test_demonstrations


def save_image(img, path):
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def save_images(images, format_str, prefix=""):
    for idx, img in enumerate(images):
        save_image(img, format_str.format(prefix, idx))


def save_images_and_tip_velocities(images, demonstration_num, tip_positions, tip_velocities, tip_velocity_file,
                                   metadata_file, crop_pixels, rotations, rotations_file, relative_target_positions,
                                   relative_target_orientations, distractor_positions, augment_to_steps=-1):
    format_str = "{}image{}.png"
    prefix = str(demonstration_num)

    assert len(images) == len(tip_positions) == len(tip_velocities) == len(crop_pixels) == len(rotations) \
           == len(relative_target_positions) == len(relative_target_orientations)

    if augment_to_steps != -1:
        # we want every demonstration to be padded to augment_to_steps
        # with the last image, and last tip velocity and rotation should be zero vectors
        # everything has to be padded
        assert augment_to_steps >= len(tip_velocities)
        pad_amount = augment_to_steps - len(tip_velocities)
        for _ in range(pad_amount):
            images.append(images[-1])
            tip_positions.append(tip_positions[-1])
            tip_velocities.append(tip_velocities[-1])
            crop_pixels.append(crop_pixels[-1])
            rotations.append(rotations[-1])
            relative_target_positions.append(relative_target_positions[-1])
            relative_target_orientations.append(relative_target_orientations[-1])

    # save images
    save_images(images=images, format_str=format_str, prefix=prefix)

    # write tip velocities
    add_to_csv_file(format_str, prefix, tip_velocities, tip_velocity_file)

    # write rotations
    add_to_csv_file(format_str, prefix, rotations, rotations_file)

    if not os.path.exists(metadata_file):
        with open(metadata_file, "x"):
            pass

    with open(metadata_file, "r") as metadata_json:
        content = metadata_json.read()
        if not content:
            content = "{}"
        data = json.loads(content)

    with open(metadata_file, "w+") as metadata_json:
        if "demonstrations" not in data:
            data["demonstrations"] = {}

        start = data["demonstrations"][str(demonstration_num - 1)]["end"] + 1 if str(demonstration_num - 1) in data[
            "demonstrations"] else 0

        data["demonstrations"][demonstration_num] = dict(
            num_tip_velocities=len(tip_velocities),
            start=start,
            end=start + len(tip_velocities) - 1,
            tip_positions=tip_positions,
            crop_pixels=crop_pixels,
            relative_target_positions=relative_target_positions,
            relative_target_orientations=relative_target_orientations,
            distractor_positions=distractor_positions
        )

        if "num_demonstrations" not in data:
            data["num_demonstrations"] = 0
        data["num_demonstrations"] += 1
        metadata_json.write(json.dumps(data))


def add_to_csv_file(format_str, prefix, content, file):
    with open(file, "a", newline='') as f:
        writer = csv.writer(f)
        for idx, data in enumerate(content):
            writer.writerow([format_str.format(prefix, idx), *data])


def get_loss(loss_params):
    if loss_params is not None:
        if loss_params == "composite":
            loss_params = {}  # for the time being set to default hyperparams
        elif loss_params == "compositeV1":
            loss_params = {"mse_lambda": 1.0, "l1_lambda": 0.0, "alignment_lambda": 0.1}
        else:
            raise ValueError("Loss is passed as an argument, but it's not composite.")

    return loss_params
