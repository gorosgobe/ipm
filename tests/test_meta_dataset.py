import unittest

import numpy as np

from lib.dataset import ImageTipVelocitiesDataset
from lib.meta_dataset import MetaTipVelocityDataset, DatasetType
from lib.utils import get_demonstrations


class MetaDatasetTest(unittest.TestCase):
    NUM_TRAINING_PER_DEMONSTRATION = 20
    NUM_TEST_PER_DEMONSTRATION = 5

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

        # make sure the ones obtained from the "train" subset are part of the original demonstration
        for x in subset:
            comparisons = list(map(lambda y: np.allclose(x["image"], y), first_real_demonstration_images))
            self.assertTrue(comparisons.count(True) == 1)

    def test_correct_data_in_training(self):
        training_meta_dataset = MetaTipVelocityDataset(
            demonstration_dataset=self.dataset,
            split=[0.8, 0.1, 0.1],
            dataset_type=DatasetType.TRAIN,
            num_training_per_demonstration=self.NUM_TRAINING_PER_DEMONSTRATION,
            num_test_per_demonstration=self.NUM_TEST_PER_DEMONSTRATION
        )

        first_demonstration = training_meta_dataset[0]
        train_first_demonstration = first_demonstration["train"]
        self.assertEqual(len(train_first_demonstration), self.NUM_TRAINING_PER_DEMONSTRATION)
        test_first_demonstration = first_demonstration["test"]
        self.assertEqual(len(test_first_demonstration), self.NUM_TEST_PER_DEMONSTRATION)

        self.check_belong_to_global_indexes(0, train_first_demonstration)
        self.check_belong_to_global_indexes(0, test_first_demonstration)

    def test_correct_data_validation(self):
        validation_meta_dataset = MetaTipVelocityDataset(
            demonstration_dataset=self.dataset,
            split=[0.8, 0.1, 0.1],
            dataset_type=DatasetType.VAL,
            num_training_per_demonstration=self.NUM_TRAINING_PER_DEMONSTRATION,
            num_test_per_demonstration=self.NUM_TEST_PER_DEMONSTRATION
        )

        validation_demonstration = validation_meta_dataset[10]  # demonstration number 130 = 120 + 10
        val_train_demonstration = validation_demonstration["train"]
        self.assertEqual(len(val_train_demonstration), self.NUM_TRAINING_PER_DEMONSTRATION)
        val_test_demonstration = validation_demonstration["test"]
        self.assertEqual(len(val_test_demonstration), self.NUM_TEST_PER_DEMONSTRATION)

        self.check_belong_to_global_indexes(130, val_train_demonstration)
        self.check_belong_to_global_indexes(130, val_test_demonstration)


if __name__ == '__main__':
    unittest.main()
