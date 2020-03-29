import enum
from collections import OrderedDict

import numpy as np
from torchmeta.transforms import ClassSplitter
from torchmeta.utils.data import MetaDataset, Task, SubsetTask

"""
Based on: 
One-Shot Visual Imitation Learning via Meta-Learning, Finn et al., 2017
"""


class DatasetType(enum.Enum):
    TRAIN = ("train", 0)
    VAL = ("val", 1)
    TEST = ("test", 2)


class MILTipVelocityDataset(MetaDataset):
    def __init__(self, demonstration_dataset, split, dataset_type, random_crop_transform=None, shuffle_within_demonstration=True,
                 random_provider=np.random.choice):
        # split is list of split values. Passed in as [train_proportion, val_proportion, test_proportion]
        self.dataset_type_str, self.dataset_type_idx = dataset_type.value
        super().__init__(meta_split=self.dataset_type_str)
        if len(split) != 3:
            raise ValueError("Split must contain proportion for train, val and test sets (3 values with sum = 1)")

        self.demonstration_dataset = demonstration_dataset
        self.split = split
        self.num_demonstrations_in_split = self.get_number_of_demonstrations()
        self.random_crop_transform = random_crop_transform
        self.shuffle_within_demonstration = shuffle_within_demonstration  # False for testing, True in general: we want to shuffle the demonstration data
        # for testing purposes
        self.random_provider = random_provider

    def __len__(self):
        # TODO: change, task is a pair of random demonstrations, so this could be set arbitrarily?
        # TODO: change below
        return self.num_demonstrations_in_split

    def get_number_of_demonstrations(self):
        # return number of demonstrations for train/val/test set depending on dataset type
        return int(self.demonstration_dataset.get_num_demonstrations() * self.split[self.dataset_type_idx])

    def get_global_demonstration_index(self, index):
        return int(
            sum(self.split[:self.dataset_type_idx]) * self.demonstration_dataset.get_num_demonstrations()
        ) + index

    def __getitem__(self, index):
        # choose two random demonstration indices
        first, second = self.random_provider(self.num_demonstrations_in_split, 2)
        assert 0 <= first < self.num_demonstrations_in_split and 0 <= second < self.num_demonstrations_in_split
        # compute global demonstration index
        first_demonstration_index = self.get_global_demonstration_index(first)
        first_start, first_end = self.demonstration_dataset.get_indices_for_demonstration(first_demonstration_index)
        train_task = MILTipVelocityDemonstrationTask(index, first_start, first_end, self.demonstration_dataset)

        second_demonstration_index = self.get_global_demonstration_index(second)
        second_start, second_end = self.demonstration_dataset.get_indices_for_demonstration(second_demonstration_index)
        test_task = MILTipVelocityDemonstrationTask(index, second_start, second_end, self.demonstration_dataset)

        first_range = np.arange(first_start, first_end + 1)
        second_range = np.arange(second_start, second_end + 1)
        # shuffling and picking random samples of the minimum size, so demonstrations are of the same size
        if self.shuffle_within_demonstration:
            minimum_dem_len = self.demonstration_dataset.get_minimum_demonstration_length()
            first_range = np.random.choice(first_range, size=minimum_dem_len, replace=False)
            np.random.shuffle(first_range)
            second_range = np.random.choice(second_range, size=minimum_dem_len, replace=False)
            np.random.shuffle(second_range)

        return OrderedDict([
            ("train", SubsetTask(train_task, first_range)),
            ("test", SubsetTask(test_task, second_range))
        ])


class MILTipVelocityDemonstrationTask(Task):
    def __init__(
            self,
            index,
            demonstration_start_index,
            demonstration_end_index,
            dataset,
            transforms=None
    ):
        super().__init__(index=index, num_classes=None)  # Regression task
        self.dataset = dataset
        self.demonstration_length = demonstration_end_index - demonstration_start_index + 1
        self.transforms = transforms

    def __len__(self):
        return self.demonstration_length

    def __getitem__(self, index):
        image_and_target_dict = self.dataset[index]
        images = image_and_target_dict["image"]
        velocities = image_and_target_dict["tip_velocities"]
        rotations = image_and_target_dict["rotations"]
        if self.transforms is not None:
            images = self.transforms(images)

        targets = np.concatenate((velocities, rotations))
        return images, targets
