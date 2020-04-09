import unittest

import numpy as np
import torch

from lib.rl.demonstration_env import SingleDemonstrationEnv


def get_mock_torch_image(idx):
    return torch.tensor([
        [
            [idx, idx + 1, idx + 2, idx + 3],
            [idx + 4, idx + 5, idx + 6, idx + 7]
        ]
    ])


def get_mock_numpy_image(idx):
    return np.array([
        [[idx], [idx + 1], [idx + 2], [idx + 3]],
        [[idx + 4], [idx + 5], [idx + 6], [idx + 7]]
    ])


IMAGE_CENTER = [1, 0]


class MockRandomChoice(object):
    def __init__(self, indices):
        self.idx = -1
        self.indices = indices

    def __call__(self, *args, **kwargs):
        self.idx += 1
        return self.indices[self.idx]


class MockDataset(object):
    def __init__(self, images, dem_to_indexes_map):
        self.images = images
        self.dem_to_indexes_map = dem_to_indexes_map

    def __getitem__(self, idx):
        return dict(image=self.images[idx], tip_velocities=np.array([-1.0, 0.0, 1.0]),
                    rotations=np.array([-1.0, 0.0, 1.0]))

    def __len__(self):
        return len(self.images)

    def get_num_demonstrations(self):
        return len(self.dem_to_indexes_map)

    def get_indices_for_demonstration(self, dem_idx):
        return self.dem_to_indexes_map[dem_idx]


REWARD_TEST = 0.1234
NUM_EPOCHS_TRAINED = 15


class MockEstimator(object):
    def __init__(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def get_best_val_loss(self):
        return REWARD_TEST

    def get_num_epochs_trained(self):
        return NUM_EPOCHS_TRAINED


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.config = dict(
            size=(4, 2),
            cropped_size=(2, 2),
            learning_rate=0.0001,
            network_klass=None,
            seed=2019,
            num_workers=0,  # number of workers to compute RL reward
            split=[1.0, 0.0, 0.0],
            patience=3,  # smaller, need to train faster
            max_epochs=50,
            validate_epochs=1,
            device=None
        )

    @staticmethod
    def check_observation_image(obs, idx):
        image = obs[2:]
        image = image.reshape(2, 4, 1)
        np.testing.assert_array_equal(get_mock_numpy_image(idx), image)

    def test_demonstrations_sampled_and_center_crop_applied_to_start_of_demonstrations(self):
        mock_random_choice = MockRandomChoice([(0, 1)])
        demonstrations = [(0, 9), (10, 16)]
        dataset = MockDataset([get_mock_torch_image(i) for i in range(17)], demonstrations)

        env = SingleDemonstrationEnv(
            demonstration_dataset=dataset,
            config=self.config,
            random_provider=mock_random_choice
        )
        obs = env.reset()
        self.check_observation_image(obs, 0)
        crop = obs[:2]
        np.testing.assert_array_equal(crop, np.array(IMAGE_CENTER))

        for i in range(1, 10):
            obs, reward, done, info = env.step([1, 1])
            self.check_observation_image(obs, i)
            crop = obs[:2]
            self.assertTrue(np.any(np.not_equal(crop, np.array(IMAGE_CENTER))))
            self.assertEqual(reward, 0)
            self.assertFalse(done)

        obs, reward, done, info = env.step([1, 1])
        self.check_observation_image(obs, 10)
        crop = obs[:2]
        np.testing.assert_array_equal(crop, np.array(IMAGE_CENTER))
        self.assertEqual(reward, 0)
        self.assertFalse(done)

    def test_correct_info_crop_is_returned(self):
        pass

    def estimator_receives_correct_training_and_validation(self):
        pass


if __name__ == '__main__':
    unittest.main()
