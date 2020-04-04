from abc import ABC

import gym
import numpy as np
import torchvision
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader

from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import FromListsDataset
from lib.rl.state import State
from lib.cv.tip_velocity_estimator import TipVelocityEstimator

"""
Main problem we have is training on a single sample at a time does not give a good estimate of the loss (reward).
Thus, we need to train on multiple demonstration images at the same time. However, the OpenAI gym environment only
allows us to return one observation at a time - by using a vectorized environment, we can use multiple environments that in
reality act like a single one. This allows us to return the same reward in all observations (negative validation loss 
when trained on crops produced by the RL agent). To train on N images, we need N Slave environments.
"""


class SpaceProviderEnv(gym.Env, ABC):
    def __init__(self, image_size):
        super().__init__()
        width, height = image_size
        self.action_space = gym.spaces.MultiDiscrete([width * 2 - 1, height * 2 - 1])
        image_size_1d = width * height * 3
        eps = 1e-4
        image_lower_bound_1d = np.full((image_size_1d,), -1.0 - eps)
        image_upper_bound_1d = np.full((image_size_1d,), 1.0 + eps)
        low = np.concatenate((np.array([-width - eps, -height - eps]), image_lower_bound_1d))
        high = np.concatenate((np.array([width + eps, height + eps]), image_upper_bound_1d))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def render(self, mode='human'):
        raise NotImplementedError("Environment is not renderable :(")
        # Without pass, PyCharm thinks the method has not been implemented, even if the implementation is to raise the exception
        pass


class SingleDemonstrationEnv(SpaceProviderEnv):
    def __init__(self, demonstration_dataset, config, random_provider=np.random.choice, estimator=TipVelocityEstimator):
        super().__init__(config["size"])
        self.demonstration_dataset = demonstration_dataset
        self.config = config
        self.random_provider = random_provider
        self.training_split = config["split"][0]
        self.estimator = estimator
        self.cropped_width, self.cropped_height = self.config["cropped_size"]
        self.width, self.height = self.config["size"]
        self.pixel_cropper = TrainingPixelROI(self.cropped_height, self.cropped_width)
        self.to_tensor = torchvision.transforms.ToTensor()

        self.demonstration_states = []
        self.state = None
        self.next_state = None
        self.start = None
        self.demonstration_img_idx = None
        self.end = None

        self.epoch_list = []

    def reset(self):
        # sample new demonstration
        demonstration_idx = self.random_provider(
            int(self.training_split * self.demonstration_dataset.get_num_demonstrations()))
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
        self.state = self.next_state
        return self.state.get() if not done else None, reward, done, {}

    def done(self):
        return len(self.demonstration_states) == self.end - self.start + 1

    def apply_action(self, action):
        action = np.array(action) - np.array([self.width - 1, self.height - 1])
        self.demonstration_img_idx += 1
        if self.done():
            return self.state.apply_action(None, action[0], action[1]), True

        new_state = self.state.apply_action(self.demonstration_dataset[self.demonstration_img_idx], action[0],
                                            action[1])
        return new_state, False

    def get_reward(self, done):
        if not done:
            return 0
        # all images are in self.demonstration_states, train with those
        # last demonstration state is dummy state to hold last crop
        assert len(self.demonstration_states) == self.end - self.start + 2
        cropped_images_and_bounding_boxes = [
            self.pixel_cropper.crop(self.demonstration_states[i].get_np_image(), self.demonstration_states[i + 1].get_center_crop()) for i
            in range(len(self.demonstration_states) - 1)
        ]
        cropped_images = map(lambda img_n_box: img_n_box[0], cropped_images_and_bounding_boxes)
        cropped_images = list(map(lambda img: self.to_tensor(img), cropped_images))
        tip_velocities = [state.get_tip_velocity() for state in self.demonstration_states[:-1]]
        rotations = [state.get_rotations() for state in self.demonstration_states[:-1]]
        training_dataset, validation_dataset = FromListsDataset(cropped_images, tip_velocities, rotations).shuffle().split()
        # train with everything as the batch, its small anyways
        train_data_loader = DataLoader(
            training_dataset,
            batch_size=len(training_dataset),
            num_workers=2,
            shuffle=True
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=len(validation_dataset),
            num_workers=2,
            shuffle=True
        )
        estimator = self.estimator(
            batch_size=len(training_dataset),
            learning_rate=self.config["learning_rate"],
            image_size=self.config["cropped_size"],  # learning from cropped size
            network_klass=self.config["network_klass"],
            device=self.config["device"],
            patience=self.config["patience"],
            verbose=False
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
