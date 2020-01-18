from lib.camera_robot import CameraRobot
from lib.controller import TipVelocityController, IdentityCropper
from lib.scenes import CameraTextureReachCubeScene
import json
import time

if __name__ == "__main__":

    with CameraTextureReachCubeScene(headless=True) as pr:
        camera_robot = CameraRobot(pr)
        model_name = "M2"
        controller = TipVelocityController("models/{}.pt".format(model_name), IdentityCropper())
        print(controller.get_model().test_loss)

        test_name = "{}-{}.test".format(model_name, int(time.time()))
        min_distances = {}

        with open("test_offsets.json", "r") as f:
            content = f.read()
            json_angles_list = json.loads(content)["angles"]

        achieved_count = 0
        for idx in range(len(json_angles_list)):
            offset = 0
            try:
                images, achieved, min_distance = camera_robot.run_controller_simulation(controller, offset)
                min_distances[str(idx)] = min_distance
            except:
                achieved = False

            if achieved:
                achieved_count += 1
            else:
                continue
            print("Achieved: ", achieved_count / (idx + 1), "{}/{}".format(achieved_count, idx + 1))

        print("Achieved: ", achieved_count / len(json_angles_list))

        with open(test_name, "w") as f:
            f.write(json.dumps(min_distances))
