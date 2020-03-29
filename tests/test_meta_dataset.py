import unittest

import numpy as np

from lib.dataset import ImageTipVelocitiesDataset
from lib.meta.meta_dataset import MILTipVelocityDataset, DatasetType


# Simulates a controllable np.random.choice(x, 2) execution
class MockRandomChoice(object):
    def __init__(self, first_idx, second_idx):
        self.first_idx = first_idx
        self.second_idx = second_idx

    def __call__(self, *args, **kwargs):
        return self.first_idx, self.second_idx


class MetaDatasetTest(unittest.TestCase):

    def setUp(self):
        dataset = "../scene1/scene1"
        self.dataset = ImageTipVelocitiesDataset(
            velocities_csv=f"{dataset}/velocities.csv",
            rotations_csv=f"{dataset}/rotations.csv",
            metadata=f"{dataset}/metadata.json",
            root_dir=dataset
        )

    def check_belong_to_global_indexes(self, demonstration_idx, subset):
        start, end = self.dataset.get_indices_for_demonstration(demonstration_idx)
        first_real_demonstration_images = []
        for i in range(start, end + 1):
            first_real_demonstration_images.append(self.dataset[i]["image"])

        # make sure the ones obtained from the subset are part of the original demonstration
        for x in subset:
            image, target = x
            comparisons = list(map(lambda y: np.allclose(image, y), first_real_demonstration_images))
            self.assertTrue(comparisons.count(True) == 1)

    def test_correct_data_in_training(self):
        training_meta_dataset = MILTipVelocityDataset(
            demonstration_dataset=self.dataset,
            split=[0.8, 0.1, 0.1],
            dataset_type=DatasetType.TRAIN,
            shuffle_within_demonstration=False,
            random_provider=MockRandomChoice(0, 1)
        )

        first_demonstration = training_meta_dataset[-1]  # index ignored, as they are sampled
        train_first_demonstration = first_demonstration["train"]
        self.assertEqual(len(train_first_demonstration), 34)
        test_first_demonstration = first_demonstration["test"]
        self.assertEqual(len(test_first_demonstration), 30)

        self.check_belong_to_global_indexes(0, train_first_demonstration)
        self.check_belong_to_global_indexes(1, test_first_demonstration)

    def test_correct_data_validation(self):
        validation_meta_dataset = MILTipVelocityDataset(
            demonstration_dataset=self.dataset,
            split=[0.8, 0.1, 0.1],
            dataset_type=DatasetType.VAL,
            shuffle_within_demonstration=False,
            random_provider=MockRandomChoice(10, 14)
        )

        validation_demonstration = validation_meta_dataset[-1]  # index ignored, as they are sampled
        val_train_demonstration = validation_demonstration["train"]
        self.assertEqual(len(val_train_demonstration), 36)  # demonstration 120 + 10 = 130 has length 36
        val_test_demonstration = validation_demonstration["test"]
        self.assertEqual(len(val_test_demonstration), 38)  # last demonstration 120 + 14 = 134 has length 30

        self.check_belong_to_global_indexes(130, val_train_demonstration)
        self.check_belong_to_global_indexes(134, val_test_demonstration)

    def test_shuffle_picks_subset(self):
        training_meta_dataset = MILTipVelocityDataset(
            demonstration_dataset=self.dataset,
            split=[0.8, 0.1, 0.1],
            dataset_type=DatasetType.TRAIN,
            shuffle_within_demonstration=True,
            random_provider=MockRandomChoice(0, 1)
        )

        demonstration = training_meta_dataset[-1]
        train_demonstration = demonstration["train"]
        self.assertEqual(len(train_demonstration), self.dataset.get_minimum_demonstration_length())
        test_demonstration = demonstration["test"]
        self.assertEqual(len(test_demonstration), self.dataset.get_minimum_demonstration_length())

        self.check_belong_to_global_indexes(0, train_demonstration)
        self.check_belong_to_global_indexes(1, test_demonstration)


if __name__ == '__main__':
    unittest.main()
