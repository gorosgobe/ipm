import enum

from torchmeta.transforms import ClassSplitter
from torchmeta.utils.data import MetaDataset, Task


class DatasetType(enum.Enum):
    TRAIN = ("train", 0)
    VAL = ("val", 1)
    TEST = ("test", 2)


class MetaTipVelocityDataset(MetaDataset):
    def __init__(self, demonstration_dataset, split, dataset_type, num_training_per_demonstration,
                 num_test_per_demonstration):
        # split is list of split values. Passed in as [train_proportion, val_proportion, test_proportion]
        self.dataset_type_str, self.dataset_type_idx = dataset_type.value
        super().__init__(
            meta_split=self.dataset_type_str,
            dataset_transform=ClassSplitter(
                num_train_per_class=num_training_per_demonstration,
                num_test_per_class=num_test_per_demonstration)
        )
        if len(split) != 3:
            raise ValueError("Split must contain proportion for train, val and test sets (3 values with sum = 1)")
        self.demonstration_dataset = demonstration_dataset
        self.split = split
        self.num_training_per_demonstration = num_training_per_demonstration
        self.num_test_per_demonstration = num_test_per_demonstration

    def __len__(self):
        # every demonstration is considered a single task
        # return number of demonstrations for train/val/test set depending on dataset type
        return int(self.demonstration_dataset.get_num_demonstrations() * self.split[self.dataset_type_idx])

    def __getitem__(self, index):
        # compute global demonstration index
        demonstration_index = int(
            sum(self.split[:self.dataset_type_idx]) * self.demonstration_dataset.get_num_demonstrations()
        ) + index
        start, end = self.demonstration_dataset.get_indices_for_demonstration(demonstration_index)
        task = MetaTipVelocityTask(index, start, end, self.demonstration_dataset)  # task is global
        if self.dataset_transform is not None:
            task = self.dataset_transform(task)
        return task


class MetaTipVelocityTask(Task):
    def __init__(
            self,
            index,
            demonstration_start_index,  # local to dataset
            demonstration_end_index,  # local to dataset
            dataset
    ):
        super(MetaTipVelocityTask, self).__init__(index, num_classes=None)  # Regression task
        self.demonstration_start_index = demonstration_start_index
        self.demonstration_end_index = demonstration_end_index
        self.dataset = dataset

    def get_start(self):
        return self.demonstration_start_index

    def get_end(self):
        return self.demonstration_end_index

    def __len__(self):
        return self.demonstration_end_index - self.demonstration_start_index + 1

    def __getitem__(self, index):
        assert self.demonstration_start_index + index <= self.demonstration_end_index
        return self.dataset[self.demonstration_start_index + index]
