from abc import ABC

import gym
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import FromListsDataset
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.rl.state import State


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


class SingleDemonstrationEnv(SpaceProviderEnv):
    def __init__(self, demonstration_dataset, config, random_provider=np.random.choice, estimator=TipVelocityEstimator,
                 use_split_idx=0, skip_reward=False):
        super().__init__(config["size"])
        self.demonstration_dataset = demonstration_dataset
        self.config = config
        self.random_provider = random_provider
        # TODO: use this properly
        self.training_split = config["split"][use_split_idx]  # 0 for training, 1 for validation, 2 for test
        self.estimator = estimator
        self.cropped_width, self.cropped_height = self.config["cropped_size"]
        self.width, self.height = self.config["size"]
        self.pixel_cropper = TrainingPixelROI(self.cropped_height, self.cropped_width, add_spatial_maps=True)
        self.to_tensor = torchvision.transforms.ToTensor()

        self.demonstration_states = []
        self.state = None
        self.next_state = None
        self.start = None
        self.demonstration_img_idx = None
        self.end = None
        self.next_demonstration_idx = None

        self.epoch_list = []
        self.skip_reward = skip_reward  # skip reward computation when testing

    def set_next_demonstration(self, idx):
        # to manually set the next demonstration and not rely on random demonstrations
        self.next_demonstration_idx = idx

    def reset(self):
        # sample new demonstration
        if self.next_demonstration_idx is None:
            # TODO: fix training split so it picks it from the right place
            demonstration_idx = self.random_provider(
                int(self.training_split * self.demonstration_dataset.get_num_demonstrations()))
        else:
            demonstration_idx = self.next_demonstration_idx
            self.next_demonstration_idx = None  # reset, set_next_demonstration needs to be called before every reset
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

    def get_reward(self, done):
        if self.skip_reward or not done:
            return 0
        # all images are in self.demonstration_states, train with those
        # last demonstration state is dummy state to hold last crop
        assert len(self.demonstration_states) == self.end - self.start + 2
        cropped_images_and_bounding_boxes = [
            self.pixel_cropper.crop(
                self.demonstration_states[i].get_np_image(),
                self.demonstration_states[i + 1].get_center_crop()
            ) for i in range(len(self.demonstration_states) - 1)
        ]
        cropped_images = map(lambda img_n_box: img_n_box[0], cropped_images_and_bounding_boxes)
        cropped_images = list(map(lambda img: self.to_tensor(img), cropped_images))
        tip_velocities = [state.get_tip_velocity() for state in self.demonstration_states[:-1]]
        rotations = [state.get_rotations() for state in self.demonstration_states[:-1]]
        training_dataset, validation_dataset = FromListsDataset(cropped_images, tip_velocities,
                                                                rotations).shuffle().split()
        # train with everything as the batch, its small anyways
        train_data_loader = DataLoader(
            training_dataset,
            batch_size=len(training_dataset),
            num_workers=self.config["num_workers"],
            shuffle=True
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=len(validation_dataset),
            num_workers=self.config["num_workers"],
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

    def get_epoch_list(self):
        return self.epoch_list
