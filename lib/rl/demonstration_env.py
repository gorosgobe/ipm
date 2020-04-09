from abc import ABC

import gym
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import FromListsDataset
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.rl.state import State
from lib.rl.utils import CropTester, CropTestModality
from lib.common.test_utils import get_distance_between_boxes


class SpaceProviderEnv(gym.Env, ABC):
    def __init__(self, image_size):
        super().__init__()
        width, height = image_size
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
        image_size_1d = width * height * 3
        image_lower_bound_1d = np.full((image_size_1d,), -1.0)
        image_upper_bound_1d = np.full((image_size_1d,), 1.0)
        low = np.concatenate((np.array([-width, -height]), image_lower_bound_1d))
        self.dummy_observation = low
        high = np.concatenate((np.array([width, height]), image_upper_bound_1d))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def render(self, mode='human'):
        raise NotImplementedError("Environment is not renderable... yet :(")
        # Without pass, PyCharm thinks the method has not been implemented, even if the implementation is to raise the exception
        pass


class TestRewardSingleDemonstrationEnv(SpaceProviderEnv):
    def __init__(self, demonstration_dataset, config, random_provider=np.random.choice, estimator=TipVelocityEstimator,
                 use_split_idx=0, **_kwargs):
        super().__init__(config["size"])
        self.demonstration_dataset = demonstration_dataset
        self.config = config
        self.random_provider = random_provider
        self.training_split = config["split"][use_split_idx]  # 0 for training, 1 for validation, 2 for test
        self.estimator = estimator
        self.cropped_width, self.cropped_height = self.config["cropped_size"]
        self.width, self.height = self.config["size"]

        self.demonstration_states = []
        self.state = None
        self.next_state = None
        self.start = None
        self.demonstration_img_idx = None
        self.end = None

        # ----
        # Use a simpler, faster reward to test the agent is learning something
        # In this case, we will use the negative distance between the predicted and the expected crops
        # Compared to our validation loss based reward, this is not sparse and is very fast to compute
        self.crop_tester = CropTester(config, demonstration_dataset, CropTestModality.TRAINING.value, self.__class__)

    def reset(self):
        # sample new demonstration
        demonstration_idx, val_demonstration_idx = self.random_provider(
            int(self.training_split * self.demonstration_dataset.get_num_demonstrations()), size=2)
        self.start, self.end = self.demonstration_dataset.get_indices_for_demonstration(demonstration_idx)
        self.demonstration_img_idx = self.start
        self.state = State(self.demonstration_dataset[self.start])
        self.next_state = None
        self.demonstration_states = [self.state]
        return self.state.get()

    def step(self, action):
        self.next_state, done = self.apply_action(action)
        # if done is True, still add to record last crop (dummy state with image = None)
        self.demonstration_states.append(self.next_state)
        reward = self.get_reward(done)
        center_crop_pixel = self.next_state.get_center_crop()
        self.state = self.next_state
        return self.state.get() if not done else self.dummy_observation, reward, done, dict(
            center_crop_pixel=center_crop_pixel)

    def done(self):
        return len(self.demonstration_states) == self.end - self.start + 1

    def apply_action(self, action):
        self.demonstration_img_idx += 1
        if self.done():
            return self.state.apply_action(None, action[0], action[1], self.cropped_width, self.cropped_height), True

        new_state = self.state.apply_action(self.demonstration_dataset[self.demonstration_img_idx], action[0],
                                            action[1], self.cropped_width, self.cropped_height)
        return new_state, False

    def get_reward(self, _done):
        distance_between_crops, _, _, _, _ = self.crop_tester.get_score(
            criterion=get_distance_between_boxes,
            gt_demonstration_idx=self.demonstration_img_idx - 1,  # index has advanced to next one
            width=self.width,
            height=self.height,
            predicted_center=self.next_state.get_center_crop(),
            cropped_width=self.cropped_width,
            cropped_height=self.cropped_height
        )
        return -distance_between_crops

    def get_epoch_list_stats(self):
        raise NotImplementedError("No estimators are trained")


class SingleDemonstrationEnv(SpaceProviderEnv):
    def __init__(self, demonstration_dataset, config, random_provider=np.random.choice, estimator=TipVelocityEstimator,
                 use_split_idx=0, skip_reward=False):
        super().__init__(config["size"])
        self.demonstration_dataset = demonstration_dataset
        self.config = config
        self.random_provider = random_provider
        self.split_coeff = sum(config["split"][:use_split_idx])  # 0 for training, 1 for validation, 2 for test
        self.estimator = estimator
        self.cropped_width, self.cropped_height = self.config["cropped_size"]
        self.width, self.height = self.config["size"]
        self.pixel_cropper = TrainingPixelROI(self.cropped_height, self.cropped_width, add_spatial_maps=True)
        self.to_tensor = torchvision.transforms.ToTensor()

        self.demonstration_states = []
        self.state = None
        # Note: both training and validation demonstrations come from the training set: these are training and validation
        # for the model that will produce the reward for the agent. The agent still has to produce crops for both sets.
        # training demonstration
        self.start = None
        self.end = None
        # validation demonstration
        self.val_start = None
        self.val_end = None
        # indices
        self.indices = None
        # current pointer, points to training or validation demonstration in indices
        self.curr_demonstration_img_idx = None
        # store the final training demonstration's crop to be able to apply it when computing the rewards
        self.final_training_crop = None

        self.epoch_list = []
        self.skip_reward = skip_reward  # skip reward computation when testing

    def reset(self):
        self.curr_demonstration_img_idx = 0
        self.final_training_crop = None
        # sample new demonstration, one for training and one for validation, both DIFFERENT (replace = False)
        # but within correct split
        demonstration_idx, val_demonstration_idx = self.random_provider(
            int(self.split_coeff * self.demonstration_dataset.get_num_demonstrations()), size=2, replace=False)
        # training demonstration
        self.start, self.end = self.demonstration_dataset.get_indices_for_demonstration(demonstration_idx)
        # validation demonstration
        self.val_start, self.val_end = self.demonstration_dataset.get_indices_for_demonstration(val_demonstration_idx)
        self.indices = list(range(self.start, self.end + 1)) + list(range(self.val_start, self.val_end + 1))
        # state
        self.state = State(self.get_curr_demonstration_data())
        self.demonstration_states = [self.state]
        return self.state.get()

    def get_curr_demonstration_data(self):
        return self.demonstration_dataset[self.get_curr_demonstration_idx()]

    def get_curr_demonstration_idx(self):
        assert self.curr_demonstration_img_idx in range(len(self.indices))
        return self.indices[self.curr_demonstration_img_idx]

    def step(self, action):
        next_state, done = self.apply_action(action)
        center_crop_pixel = next_state.get_center_crop()
        if self.training_demonstration_done():
            # we want to return to the agent the first validation state, with center crop, but want the reward function
            # to see the previous crop (reward func only uses list of demonstration_states)
            # next state has validation image and center crop
            self.demonstration_states.append(State(next_state.get_data(), *self.final_training_crop))
            center_crop_pixel = self.final_training_crop
        else:
            # if done is True, still add to record last crop (dummy state with image = None)
            self.demonstration_states.append(next_state)

        reward = self.get_reward(done)
        self.state = next_state
        return self.state.get() if not done else self.dummy_observation, reward, done, dict(
            center_crop_pixel=center_crop_pixel)

    def training_demonstration_done(self):
        return self.curr_demonstration_img_idx < len(self.indices) and self.get_curr_demonstration_idx() == self.val_start

    def validation_demonstration_done(self):
        return self.curr_demonstration_img_idx == len(self.indices)

    def apply_action(self, action):
        self.curr_demonstration_img_idx += 1

        if self.training_demonstration_done():
            # return starting state for validation demonstration
            # store final crop to apply to last training demonstration image
            self.final_training_crop = self.state.apply_action(
                None, action[0], action[1], self.cropped_width, self.cropped_height
            ).get_center_crop()
            return State(self.get_curr_demonstration_data()), False

        if self.validation_demonstration_done():
            return self.state.apply_action(None, action[0], action[1], self.cropped_width, self.cropped_height), True

        new_state = self.state.apply_action(self.get_curr_demonstration_data(), action[0], action[1],
                                            self.cropped_width, self.cropped_height)
        return new_state, False

    def get_reward(self, done):
        if self.skip_reward:
            return 0

        if not done:
            return 0

        # all images are in self.demonstration_states, train with those
        # last demonstration state is dummy state to hold last crop
        assert len(self.demonstration_states) == (self.end - self.start + 1) + (self.val_end - self.val_start + 1) + 1

        cropped_images_and_bounding_boxes = (
            self.pixel_cropper.crop(
                self.demonstration_states[i].get_np_image(),
                self.demonstration_states[i + 1].get_center_crop()
            ) for i in range(len(self.demonstration_states) - 1)
        )

        cropped_images = map(lambda img_n_box: img_n_box[0], cropped_images_and_bounding_boxes)
        cropped_images = list(map(lambda img: self.to_tensor(img), cropped_images))
        tip_velocities = [state.get_tip_velocity() for state in self.demonstration_states[:-1]]
        rotations = [state.get_rotations() for state in self.demonstration_states[:-1]]
        # Use first demonstration for training, second for validation (correlations within demonstrations)
        training_dataset, validation_dataset = FromListsDataset(cropped_images, tip_velocities,
                                                                rotations).split(self.end - self.start + 1)
        # train with everything as the batch, its small anyways
        train_data_loader = DataLoader(
            training_dataset,
            batch_size=len(training_dataset),
            num_workers=self.config["num_workers"],
            shuffle=self.config["shuffle"]
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=len(validation_dataset),
            num_workers=self.config["num_workers"],
            shuffle=self.config["shuffle"]
        )
        estimator = self.estimator(
            batch_size=len(training_dataset),
            learning_rate=self.config["learning_rate"],
            image_size=self.config["cropped_size"],  # learning from cropped size
            network_klass=self.config["network_klass"],
            device=self.config["device"],
            patience=self.config["patience"],
            verbose=False,
            # mainly for testing purposes, but can be used to modify the optimiser too
            optimiser_params=self.config["optimiser_params"] if "optimiser_params" in self.config else None
        )
        estimator.train(
            data_loader=train_data_loader,
            max_epochs=self.config["max_epochs"],
            validate_epochs=self.config["validate_epochs"],
            val_loader=validation_data_loader,
        )
        reward = -estimator.get_best_val_loss()
        self.epoch_list.append(estimator.get_num_epochs_trained())
        return reward

    def get_epoch_list_stats(self):
        if len(self.epoch_list) == 0:
            raise ValueError("No estimators were trained")
        arr = np.array(self.epoch_list)
        return np.mean(arr), np.std(arr)
