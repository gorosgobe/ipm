import csv
import json
import os

import cv2


def save_image(img, path):
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def save_images(images, format_str, prefix=""):
    for idx, img in enumerate(images):
        save_image(img, format_str.format(prefix, idx))

def save_images_and_tip_velocities(images, demonstration_num, tip_positions, tip_velocities, tip_velocity_file,
                                   metadata_file, crop_pixels):
    format_str = "{}image{}.png"
    prefix = str(demonstration_num)
    save_images(images=images, format_str=format_str, prefix=prefix)

    with open(tip_velocity_file, "a", newline='') as f:
        writer = csv.writer(f)
        for idx, vel in enumerate(tip_velocities):
            writer.writerow([format_str.format(prefix, idx), *vel])

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
            crop_pixels=crop_pixels
        )

        if "num_demonstrations" not in data:
            data["num_demonstrations"] = 0
        data["num_demonstrations"] += 1
        metadata_json.write(json.dumps(data))
