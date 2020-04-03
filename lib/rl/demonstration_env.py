from abc import ABC

import gym
import numpy as np
import torchvision
from stable_baselines.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader

from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import FromListsDataset
from state import State
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
        image_lower_bound_1d = np.full((image_size_1d,), -1.0)
        image_upper_bound_1d = np.full((image_size_1d,), 1.0)
        low = np.concatenate((np.array([-width, -height]), image_lower_bound_1d))
        high = np.concatenate((np.array([width, height]), image_upper_bound_1d))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def render(self, mode='human'):
        raise NotImplementedError("Environment is not renderable :(")
        # Without pass, PyCharm thinks the method has not been implemented, even if the implementation is to raise the exception
        pass


class SlaveEnv(SpaceProviderEnv):
    def __init__(self, leader, id):
        super().__init__(leader.get_image_size())
        self.leader = leader
        self.id = id
        self.demonstration_start = None
        self.demonstration_image_idx = None
        self.demonstration_end = None

    def step(self, action):
        print("step")
        return self.leader.request_step(self.id)

    def next_from_leader(self):
        self.demonstration_image_idx += 1

    def reset(self):
        print("reset")
        observation, demonstration_start, demonstration_end = self.leader.request_reset(self.id)
        self.demonstration_start = demonstration_start
        self.demonstration_image_idx = demonstration_start  # initial image in demonstration
        self.demonstration_end = demonstration_end
        return observation

    def get_current_demonstration(self):
        return self.demonstration_start, self.demonstration_image_idx, self.demonstration_end

    def are_you_done(self):
        return self.demonstration_image_idx == self.demonstration_end + 1


class LeaderEnv(SlaveEnv):
    def __init__(self, number_envs, demonstration_dataset, config, random_provider=np.random.choice, estimator=TipVelocityEstimator):
        self.config = config
        super().__init__(self, 0)
        self.number_envs = number_envs  # including leader
        self.all = [self, *[SlaveEnv(self, i) for i in range(1, self.number_envs)]]
        # training split must be the same as the one applied to the demonstration dataset
        self.training_split = self.config["split"][0]
        self.demonstration_idxs = {}
        self.slave_done = {i: True for i in range(self.number_envs)}
        self.demonstration_dataset = demonstration_dataset
        self.actions = None
        self.cropped_width, self.cropped_height = self.config["cropped_size"]
        self.width, self.height = self.config["size"]
        self.pixel_cropper = TrainingPixelROI(self.cropped_height, self.cropped_width)
        self.to_tensor = torchvision.transforms.ToTensor()
        # Cache of states of all slaves
        self.states = {}
        # Reward for this round
        self.reward = None
        # Cache of info for slaves
        self.infos = {i: {} for i in range(self.number_envs)}
        self.reward_list = []
        self.random_provider = random_provider  # for testing purposes
        self.estimator = estimator  # for testing purposes

    def request_step(self, id):
        # should have already received all actions, i.e. set_actions has already been called
        if id == 0:
            # I am the leader, perform all computations for this round
            # Are slaves done with their respective demonstrations?
            self.slave_done = {id: slave.are_you_done() for id, slave in enumerate(self.all)}
            # new observations, applying action on current observation for slaves that aren't done
            self.states = {id: self.apply_action(id) if not self.slave_done[id] else self.states[id] for id in
                           range(self.number_envs)}
            # calculate reward of applying actions on old states
            # i.e. how good is our model at predicting targets from cropped images
            self.reward = self.get_reward()

        observation = self.states[id].get()  # get the actual np array
        reward = self.reward  # reward is the same for everyone
        done = self.slave_done[id]
        info = self.infos[id]

        return observation, reward, done, info

    def request_reset(self, id):
        # sample demonstration for supplied child id, and keep track of it
        print(self.slave_done)
        print("ID:", id)
        assert self.slave_done[id]  # slave should be done
        # sample demonstration from training data
        # demonstration index is already global, as training data starts at index 0 in the dataset
        demonstration_idx = self.random_provider(
            int(self.training_split * self.demonstration_dataset.get_num_demonstrations()))
        start, end = self.demonstration_dataset.get_indices_for_demonstration(demonstration_idx)
        self.demonstration_idxs[id] = demonstration_idx
        # with first (0th) image of the demonstration
        self.states[id] = State(self.demonstration_dataset[start])
        self.slave_done[id] = False
        return self.states[id].get(), start, end

    def set_actions(self, actions):
        # sets and transforms actions into offset form
        self.actions = [action - np.array([self.width - 1, self.height - 1]) for action in actions]

    def get_all(self):
        return self.all

    def get_image_size(self):
        return self.config["size"]

    def apply_action(self, id):
        # move slaves to next
        self.all[id].next_from_leader()
        state = self.states[id]
        action = self.actions[id]
        # State object: has next image, and previous crop (i.e. previous state + action)
        # guaranteed slave is not done, so can get the next image in the dataset
        _, next_image_idx, _ = self.all[id].get_current_demonstration()
        new_state = state.apply_action(self.demonstration_dataset[next_image_idx], action[0], action[1])
        return new_state

    def get_reward(self):
        # all images are in self.states, train with those
        # TODO: image here is wrong, fix
        cropped_images_and_bounding_boxes = [
            self.pixel_cropper.crop(self.states[id].get_np_image(), self.states[id].get_center_crop()) for id in
            range(self.number_envs)
        ]
        cropped_images = map(lambda img_n_box: img_n_box[0], cropped_images_and_bounding_boxes)
        cropped_images = list(map(lambda img: self.to_tensor(img), cropped_images))
        tip_velocities = [self.states[id].get_tip_velocity() for id in range(self.number_envs)]
        rotations = [self.states[id].get_rotations() for id in range(self.number_envs)]
        training_dataset, validation_dataset = FromListsDataset(cropped_images, tip_velocities, rotations).split()
        print("Training, validation sizes", len(training_dataset), len(validation_dataset))
        # train with everything as the batch, its small anyways
        train_data_loader = DataLoader(
            training_dataset,
            batch_size=len(training_dataset),
            num_workers=0,
            shuffle=True
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=len(validation_dataset),
            num_workers=0,
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
        self.reward_list.append(reward)
        return reward


class MultipleSynchronousDemonstrationsEnv(DummyVecEnv):
    def __init__(self, n_envs, demonstration_dataset, config):
        self.leader = LeaderEnv(n_envs, demonstration_dataset, config)
        all = self.leader.get_all()
        super().__init__([(lambda env=env: lambda: env)() for env in all])

    def step_async(self, actions):
        # store all actions
        self.leader.set_actions(actions)
        return super().step_async(actions)
