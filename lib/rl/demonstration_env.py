from abc import ABC

import gym
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader

from controller import TrainingPixelROI
from dataset import FromListsDataset
from state import State
from tip_velocity_estimator import TipVelocityEstimator

"""
Main problem we have is training on a single sample at a time does not give a good estimate of the loss (reward).
Thus, we need to train on multiple demonstration images at the same time. However, the OpenAI gym environment only
allows us to return one observation at a time - by using a vectorized environment, we can use multiple environments that in
reality act like a single one. This allows us to return the same reward in all observations (negative validation loss 
when trained on crops produced by the RL agent). To train on N images, we need N Slave environments.
"""


class SpaceProviderEnv(gym.Env, ABC):
    def __init__(self):
        super().__init__()
        # TODO: define spaces
        self.action_space = gym.spaces.Box(low=-1, high=-1, shape=(1,), dtype=np.int32)
        self.observation_space = None

    def render(self, mode='human'):
        raise NotImplementedError("Environment is not renderable :(")
        # Without pass, PyCharm thinks the method has not been implemented, even if the implementation is to raise the exception
        pass


class SlaveEnv(SpaceProviderEnv):
    def __init__(self, leader, id):
        super().__init__()
        self.leader = leader
        self.id = id
        self.demonstration_start = None
        self.demonstration_image_idx = None
        self.demonstration_end = None

    def step(self, action):
        self.demonstration_image_idx += 1
        return self.leader.request_step(self.id)

    def reset(self):
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
    def __init__(self, n_envs, demonstration_dataset, config):
        super().__init__(self, 0)
        self.all = [self, *[SlaveEnv(self, i) for i in range(1, n_envs)]]
        self.n_envs = n_envs  # including leader
        self.config = config
        self.training_split = self.config["training_split"]
        self.demonstration_idxs = {}
        self.slave_done = {i: True for i in range(n_envs)}
        self.demonstration_dataset = demonstration_dataset
        self.actions = None
        width, height = self.config["size"]
        self.pixel_cropper = TrainingPixelROI(height, width)
        # Cache of states of all slaves
        self.states = {}
        # Reward for this round
        self.reward = None
        # Cache of info for slaves
        self.infos = {}

    def request_step(self, id):
        # should have already received all actions, i.e. set_actions has already been called
        if id == 0:
            # I am the leader, perform all computations for this round
            # Are slaves done with their respective demonstrations?
            self.slave_done = {id: slave.are_you_done() for id, slave in enumerate(self.all)}
            # new observations, applying action on current observation for slaves that aren't done
            self.states = {id: self.apply_action(id) for id, _ in enumerate(self.all) if not self.slave_done[id]}
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
        assert self.slave_done[id]  # slave should be done
        # sample demonstration from training data
        # demonstration index is already global, as training data starts at index 0 in the dataset
        demonstration_idx = np.random.choice(
            int(self.training_split * self.demonstration_dataset.get_num_demonstrations()))
        start, end = self.demonstration_dataset.get_indices_for_demonstration(demonstration_idx)
        self.demonstration_idxs[id] = demonstration_idx
        # with first (0th) image of the demonstration
        self.states[id] = State(self.demonstration_dataset[start])
        self.slave_done[id] = False
        return self.states[id].get(), start, end

    def set_actions(self, actions):
        self.actions = actions

    def apply_action(self, id):
        # TODO: complete
        state = self.states[id]
        action = self.actions[id]
        # State object: has next image, and previous crop (i.e. previous state + action)
        # guaranteed slave is not done, so can get the next image in the dataset
        _, next_image_idx, _ = self.all[id].get_current_demonstration()
        new_state = state.apply_action(self.demonstration_dataset[next_image_idx], action[0], action[1])
        return new_state.get() if new_state is not None else None

    def get_reward(self):
        # all images are in self.states, train with those
        cropped_images = [self.pixel_cropper.crop(self.states[id].get_np_image(), self.actions[id])[0] for id in
                          range(self.n_envs)]
        tip_velocities = [self.states[id].get_tip_velocity() for id in range(self.n_envs)]
        rotations = [self.states[id].get_rotations() for id in range(self.n_envs)]
        dataset = FromListsDataset(cropped_images, tip_velocities, rotations)
        train_data_loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True
        )
        estimator = TipVelocityEstimator(
            batch_size=self.config["batch_size"],
            learning_rate=self.config["learning_rate"],
            image_size=self.config["size"],
            network_klass=self.config["network_klass"],
            device=self.config["device"],
            patience=self.config["patience"]
        )
        estimator.train(
            data_loader=train_data_loader,
            max_epochs=self.config["max_epochs"],
            validate_epochs=self.config["validate_epochs"],
            val_loader=self.config["validation_data_loader"],
        )
        return -estimator.get_best_val_loss()


class MultipleSynchronousDemonstrationEnv(DummyVecEnv):
    def __init__(self):
        super().__init__(-1)
        self.leader = LeaderEnv(1, -1)

    def step_async(self, actions):
        # store all actions
        self.leader.set_actions(actions)
        return super().step_async(actions)
