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
    def __init__(self, check_on_train=False, *args, **kwargs):
        self.check_on_train = check_on_train

    def __call__(self, *args, **kwargs):
        return self

    def train(self, data_loader, val_loader, *args, **kwargs):
        if not self.check_on_train:
            return
        print("Checking training and validation data")
        training_data = None
        count = 0
        for data in data_loader:
            training_data = data
        images = training_data["image"]
        for i, image in enumerate(images):
            count += 1
            np.testing.assert_array_equal(get_mock_numpy_image(i)[:, 2:4], image.permute(1, 2, 0).numpy()[:, :, :1])

        validation_data = None
        for data in val_loader:
            validation_data = data
        images = validation_data["image"]
        for i, image in enumerate(images):
            np.testing.assert_array_equal(get_mock_numpy_image(count + i)[:, 2:4], image.permute(1, 2, 0).numpy()[:, :, :1])


    def get_best_val_loss(self):
        return REWARD_TEST

    def get_num_epochs_trained(self):
        return NUM_EPOCHS_TRAINED


class MockNetwork(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    @staticmethod
    def parameters():
        return -1


class MockOptim(object):
    def __init__(self, _params, _lr):
        pass


class DemonstrationEnvTest(unittest.TestCase):

    def setUp(self):
        self.config = dict(
            size=(4, 2),
            cropped_size=(2, 2),
            learning_rate=0.0001,
            network_klass=MockNetwork,
            seed=2019,
            num_workers=0,  # number of workers to compute RL reward
            split=[1.0, 0.0, 0.0],
            patience=3,  # smaller, need to train faster
            max_epochs=50,
            validate_epochs=1,
            device=None,
            optimiser_params=dict(optim=MockOptim),
            shuffle=False
        )
        self.mock_random_choice = MockRandomChoice([(0, 1)])
        demonstrations = [(0, 9), (10, 16)]
        self.dataset = MockDataset([get_mock_torch_image(i) for i in range(17)], demonstrations)

        self.env = SingleDemonstrationEnv(
            demonstration_dataset=self.dataset,
            config=self.config,
            random_provider=self.mock_random_choice,
            estimator=MockEstimator
        )

    @staticmethod
    def check_observation_image(obs, idx):
        image = obs[2:]
        image = image.reshape(2, 4, 1)
        np.testing.assert_array_equal(get_mock_numpy_image(idx), image)

    def test_demonstrations_sampled_and_center_crop_applied_to_start_of_demonstrations(self):
        obs = self.env.reset()
        self.check_observation_image(obs, 0)
        crop = obs[:2]
        np.testing.assert_array_equal(crop, np.array(IMAGE_CENTER))

        for i in range(1, 10):
            obs, reward, done, info = self.env.step([1, 1])
            self.check_observation_image(obs, i)
            crop = obs[:2]
            self.assertTrue(np.any(np.not_equal(crop, np.array(IMAGE_CENTER))))

        obs, reward, done, info = self.env.step([1, 1])
        self.check_observation_image(obs, 10)
        crop = obs[:2]
        np.testing.assert_array_equal(crop, np.array(IMAGE_CENTER))

    def test_is_done_when_validation_demonstration_is_done(self):
        _ = self.env.reset()
        for i in range(1, 17):
            _, _, done, _ = self.env.step([1, 1])
            self.assertFalse(done)

        _, _, done, _ = self.env.step([0, 0])
        self.assertTrue(done)

    def test_reward_is_provided_at_end_of_demonstration(self):
        _ = self.env.reset()
        for i in range(1, 17):
            _, reward, _, _ = self.env.step([1, 1])
            self.assertEqual(reward, 0)

        _, reward, _, _ = self.env.step([0, 0])
        self.assertEqual(reward, -REWARD_TEST)

    def check_info(self, info, x, y):
        center = info["center_crop_pixel"]
        self.assertEqual(center[0], x)
        self.assertEqual(center[1], y)

    def test_correct_info_crop_is_returned(self):
        _ = self.env.reset()
        _, _, _, info = self.env.step([1, 1])
        self.check_info(info, 2, 0)
        # crop at 2, 0
        _, _, _, info = self.env.step([-2/3, 0])
        self.check_info(info, 0, 0)
        _, _, _, info = self.env.step([1, 1])
        self.check_info(info, 2, 0)
        _, _, _, info = self.env.step([0, -2/3])
        self.check_info(info, 2, 0)
        for i in range(5, 10):
            _, _, _, info = self.env.step([1, 1])
            self.check_info(info, 2, 0)

        # observation is first from validation, but info is previous crop, so crop starting from (2, 0)
        obs, _, _, info = self.env.step([0, 0])
        self.check_observation_image(obs, 10)
        self.check_info(info, 2, 0)
        # now, crop should be starting from center crop of first validation image, so (1, 0) + (0, 0)
        _, _, _, info = self.env.step([0, 0])
        self.check_info(info, 1, 0)

    def test_estimator_receives_correct_training_and_validation(self):
        self.env = SingleDemonstrationEnv(
            demonstration_dataset=self.dataset,
            config=self.config,
            random_provider=self.mock_random_choice,
            estimator=MockEstimator(check_on_train=True)
        )
        _ = self.env.reset()
        for i in range(1, 17):
            _ = self.env.step([1, 1])
        obs, reward, done, info = self.env.step([1, 1])
        self.assertEqual(reward, -REWARD_TEST)


if __name__ == '__main__':
    unittest.main()
