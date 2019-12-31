import cv2

from controller import TipVelocityController, IdentityCropper, OffsetCropper
from sawyer_robot import SawyerRobot
from scenes import SawyerReachCubeScene, SawyerTextureReachCubeScene, SawyerTextureDistractorsReachCubeScene
import numpy as np

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
import json

if __name__ == "__main__":

    with SawyerTextureDistractorsReachCubeScene(headless=True) as pr:
        sawyer_robot = SawyerRobot(pr)
        controller = TipVelocityController("models/M1TL4.pt", IdentityCropper())
        # OffsetCropper(cropped_height=480 // 2, cropped_width=640 // 2, offset_height=25))
        print(controller.get_model().test_loss)

        with open("test_angles.json", "r") as f:
            content = f.read()
            json_angles_list = json.loads(content)["angles"]

        achieved_count = 0
        for idx in range(len(json_angles_list)):
            offset_angles_list = json_angles_list[idx]
            offset_angles = {idx + 1: angle_offset for idx, angle_offset in enumerate(offset_angles_list)}
            try:
                images, achieved = sawyer_robot.run_controller_simulation(controller, offset_angles)
            except:
                achieved = False

            if achieved:
                achieved_count += 1
            else:
                pass
                # save images obtained
                # for img_idx, image in enumerate(images):
                #     image = cv2.convertScaleAbs(image, alpha=(255.0))
                #     #cv2.imwrite(os.path.join("/home/pablo/Desktop/errors_distractors_M1TIC", "{}test{}.png".format(idx, img_idx)),
                #                 #cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print("Achieved: ", achieved_count / (idx + 1), "{}/{}".format(achieved_count, idx + 1))

        print("Achieved: ", achieved_count / len(json_angles_list))
