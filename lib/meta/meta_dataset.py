import enum

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
    def __init__(self, demonstration_dataset, split, dataset_type, shuffle_within_demonstration=True, random_provider=np.random.choice):
        # split is list of split values. Passed in as [train_proportion, val_proportion, test_proportion]
        self.dataset_type_str, self.dataset_type_idx = dataset_type.value
        super().__init__(meta_split=self.dataset_type_str)
        if len(split) != 3:
            raise ValueError("Split must contain proportion for train, val and test sets (3 values with sum = 1)")

        self.demonstration_dataset = demonstration_dataset
        self.split = split
        self.num_demonstrations_in_split = self.get_number_of_demonstrations()
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
        return int(sum(self.split[:self.dataset_type_idx]) * self.demonstration_dataset.get_num_demonstrations()) + index

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
        if self.shuffle_within_demonstration:
            np.random.shuffle(first_range)
            np.random.shuffle(second_range)

        return {"train": SubsetTask(train_task, first_range),
                "test": SubsetTask(test_task, second_range)}


class MILTipVelocityDemonstrationTask(Task):
    def __init__(
            self,
            index,
            demonstration_start_index,
            demonstration_end_index,
            dataset
    ):
        super().__init__(index=index, num_classes=None)  # Regression task
        self.dataset = dataset
        self.demonstration_length = demonstration_end_index - demonstration_start_index + 1

    def __len__(self):
        return self.demonstration_length

    def __getitem__(self, index):
        image_and_target_dict = self.dataset[index]
        return image_and_target_dict["image"], image_and_target_dict["tip_velocities"]
