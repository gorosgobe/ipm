from abc import ABC

import gym
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv

from state import State

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
        self.demonstration_length = None
        self.demonstration_image_idx = -1

    def step(self, action):
        self.demonstration_image_idx += 1
        return self.leader.request_step(self.id)

    def reset(self):
        observation, demonstration_length = self.leader.request_reset(self.id)
        self.demonstration_length = demonstration_length
        self.demonstration_image_idx = 0  # initial image in demonstration
        return observation

    def are_you_done(self):
        return self.demonstration_length == self.demonstration_image_idx


class LeaderEnv(SlaveEnv):
    def __init__(self, n_envs, demonstration_dataset):
        super().__init__(self, 0)
        self.slaves = [SlaveEnv(self, i) for i in range(1, n_envs)]  # without leader
        self.n_envs = n_envs  # including leader
        self.demonstration_idxs = {}
        self.slave_done = {i: True for i in range(n_envs)}
        self.demonstration_dataset = demonstration_dataset
        self.actions = None

        # Cache of states of all slaves
        self.states = {}
        # Reward for this round
        self.reward = None
        # Cache of info for slaves
        self.infos = {}

    def get_all_envs(self):
        return [self, *self.slaves]

    def request_step(self, id):
        # should have already received all actions, i.e. set_actions has already been called
        if id == 0:
            # I am the leader, perform all computations for this round
            # new observations, applying action on current observation
            self.states = {id: self.apply_action(id) for id in range(self.n_envs)}
            # calculate reward of applying actions on old states
            # i.e. how good is our model at predicting targets from cropped images
            self.reward = self.get_reward()
            # Are slaves done with their respective demonstrations?
            all = self.get_all_envs()
            self.slave_done = {id: all[id].are_you_done() for id in range(self.n_envs)}

        observation = self.states[id].get()  # get the actual np array
        reward = self.reward  # reward is the same for everyone
        done = self.slave_done[id]
        info = self.infos[id]

        return observation, reward, done, info

    def request_reset(self, id):
        # sample demonstration for supplied child id, and keep track of it
        assert self.slave_done[id]  # slave should be done, or has just started
        demonstration_idx = self.demonstration_dataset.sample_demonstration()  # TODO: change
        self.demonstration_idxs[id] = demonstration_idx
        # with first (0th) image of the demonstration
        self.states[id] = State(self.demonstration_dataset.get_image_from_demonstration(demonstration_idx, 0))
        self.slave_done[id] = False
        return self.states[id].get(), self.demonstration_dataset.get_length_demonstration(demonstration_idx)

    def set_actions(self, actions):
        self.actions = actions

    def apply_action(self, id):
        # TODO: complete
        demonstration_idx = self.demonstration_idxs[id]
        state = self.states[id]
        action = self.actions[id]
        new_image =
        new_observation = None
        return new_observation

    def get_reward(self):
        # TODO: complete
        # all images are in self.states, train with those
        reward = 0
        return reward


class MultipleSynchronousDemonstrationEnv(DummyVecEnv):
    def __init__(self):
        super().__init__(-1)
        self.leader = LeaderEnv(1, -1)

    def step_async(self, actions):
        # store all actions
        self.leader.set_actions(actions)
        return super().step_async(actions)
