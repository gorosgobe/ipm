import unittest

import numpy as np

from lib.common.utils import get_preprocessing_transforms
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.rl.demonstration_env import LeaderEnv
from lib.rl.state import State


class MockRandomChoice(object):
    def __init__(self, indices):
        self.idx = -1
        self.indices = indices

    def __call__(self, *args, **kwargs):
        self.idx += 1
        return self.indices[self.idx]


REWARD_TEST = 0.1234


class MockEstimator(object):
    def __init__(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def get_best_val_loss(self):
        return REWARD_TEST


class MyTestCase(unittest.TestCase):

    def setUp(self):
        dataset = "../scene1/scene1"

        self.config = dict(
            n_envs=4,
            size=(128, 96),
            cropped_size=(32, 24),
            learning_rate=0.0001,
            network_klass=None,
            seed=2019,
            velocities_csv=f"{dataset}/velocities.csv",
            rotations_csv=f"{dataset}/rotations.csv",
            metadata=f"{dataset}/metadata.json",
            root_dir=dataset,
            num_workers=0,  # number of workers to compute RL reward
            split=[0.8, 0.1, 0.1],
            patience=3,  # smaller, need to train faster
            max_epochs=50,
            validate_epochs=1,
            device=None
        )

        preprocessing_transforms, _ = get_preprocessing_transforms(self.config["size"])

        self.dataset = ImageTipVelocitiesDataset(
            velocities_csv=f"{dataset}/velocities.csv",
            rotations_csv=f"{dataset}/rotations.csv",
            metadata=f"{dataset}/metadata.json",
            root_dir=dataset,
            transform=preprocessing_transforms
        )

    def check_indexes_of_slave(self, start, idx, end, slave):
        dem_slave_start, dem_slave_idx, dem_slave_end = slave.get_current_demonstration()
        self.assertEqual(dem_slave_start, start)
        self.assertEqual(dem_slave_idx, idx)
        self.assertEqual(dem_slave_end, end)

    def check_observation_image_matches_demonstration_image(self, observation, image_idx):
        observation_image = np.reshape(observation[2:], (96, 128, 3))
        dataset_image = self.dataset[image_idx]["image"].permute(1, 2, 0).numpy()
        np.testing.assert_allclose(
            observation_image,
            dataset_image
        )

    def test_correct_observations_are_returned_and_end_of_episodes_are_handled_properly(self):
        leader = LeaderEnv(
            number_envs=2,
            demonstration_dataset=self.dataset,
            config=self.config,
            random_provider=MockRandomChoice([0, 1]),
            estimator=MockEstimator
        )
        all = leader.get_all()

        observation = all[0].reset()
        self.check_observation_image_matches_demonstration_image(observation, 0)
        observation = all[1].reset()
        self.check_observation_image_matches_demonstration_image(observation, 34)

        states = leader.states
        self.assertEqual(states[0], State(self.dataset[0]))
        self.assertEqual(states[1], State(self.dataset[34]))

        self.check_indexes_of_slave(0, 0, 33, all[0])
        self.check_indexes_of_slave(34, 34, 63, all[1])

        leader.set_actions([[127 + 1, 95 + 5], [127 - 10, 95 + 23]])

        self.check_step(1, all[0], False)  # leader step, so states for all should be updated
        states = leader.states
        self.assertEqual(states[0], State(self.dataset[1], x_center_previous=64 + 1, y_center_previous=48 + 5))
        self.assertEqual(states[1], State(self.dataset[35], x_center_previous=64 - 10, y_center_previous=48 + 23))
        self.check_indexes_of_slave(0, 1, 33, all[0])
        self.check_indexes_of_slave(34, 35, 63, all[1])

        self.check_step(35, all[1], False)
        for i in range(2, 30):
            self.check_step(i, all[0], False)
            self.check_step(34 + i, all[1], False)
        self.check_step(30, all[0], False)
        self.check_step(None, all[1], True)

    def check_step(self, image_idx, slave, done_gt):
        observation, reward, done, _info = slave.step(-1)
        if image_idx is None:
            self.assertEqual(observation, None)
        else:
            self.check_observation_image_matches_demonstration_image(observation, image_idx)
        self.assertEqual(reward, -REWARD_TEST)
        self.assertEqual(done, done_gt)


if __name__ == '__main__':
    unittest.main()
