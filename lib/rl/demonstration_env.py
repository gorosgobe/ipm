import pprint
from abc import ABC
from functools import reduce

import gym
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from lib.common.test_utils import get_distance_between_boxes
from lib.common.utils import get_network_param_if_init_from
from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import FromListsDataset
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.dsae.dsae_action_predictor import ActionPredictorManager
from lib.dsae.dsae_networks import DSAE_TargetActionPredictor
from lib.meta.mil import MetaImitationLearning
from lib.rl.state import ImageOffsetState, FilterSpatialFeatureState
from lib.rl.utils import DatasetModality, CropScorer


class ImageSpaceProviderEnv(gym.Env, ABC):
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


class TestRewardSingleDemonstrationEnv(ImageSpaceProviderEnv):
    def __init__(self, demonstration_dataset, config, random_provider=np.random.choice, use_split_idx=0, **_kwargs):
        super().__init__(config["size"])
        self.demonstration_dataset = demonstration_dataset
        self.config = config
        self.random_provider = random_provider
        self.training_split = config["split"][use_split_idx]  # 0 for training, 1 for validation, 2 for test
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
        self.crop_scorer = CropScorer(config)

    def reset(self):
        # sample new demonstration
        demonstration_idx = self.random_provider(
            int(self.training_split * self.demonstration_dataset.get_num_demonstrations())
        )
        self.start, self.end = self.demonstration_dataset.get_indices_for_demonstration(demonstration_idx)
        self.demonstration_img_idx = self.start
        self.state = ImageOffsetState(self.demonstration_dataset[self.start])
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
            return self.state.apply_action(None, action[0], action[1], self.cropped_width, self.cropped_height,
                                           restrict_crop_move=self.config["restrict_crop_move"]), True

        new_state = self.state.apply_action(self.demonstration_dataset[self.demonstration_img_idx], action[0],
                                            action[1], self.cropped_width, self.cropped_height,
                                            restrict_crop_move=self.config["restrict_crop_move"])
        return new_state, False

    def get_reward(self, _done):
        distance_between_crops, _, _, _, _ = self.crop_scorer.get_score(
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

    def save_validation_losses_list(self, path):
        raise NotImplementedError("No estimators are trained")


class DemonstrationSampler(object):
    def __init__(self, split, num_demonstrations, dataset_type_idx, random_provider=np.random.choice):
        self.split = split
        self.dataset_type_idx = dataset_type_idx
        self.num_demonstrations = num_demonstrations
        self.random_provider = random_provider

    def sample_demonstration(self, size):
        demonstration_idxs = self.random_provider(int(self.split[self.dataset_type_idx] * self.num_demonstrations), size=size, replace=False)
        demonstration_idxs = list(map(self.get_global_demonstration_index, demonstration_idxs))
        return demonstration_idxs

    def get_demonstration_indexer(self, demonstration_dataset, demonstrations):
        # sample demonstrations
        demonstration_idxs = self.sample_demonstration(size=demonstrations)
        # training demonstration
        pair_list = list(map(demonstration_dataset.get_indices_for_demonstration, demonstration_idxs))
        return DemonstrationIndexer(
            *pair_list,
            demonstration_dataset=demonstration_dataset
        )

    def sample_train_val_demonstrations(self):
        demonstration_idx, val_demonstration_idx = self.random_provider(
            int(self.split[self.dataset_type_idx] * self.num_demonstrations), size=2,
            replace=False)
        demonstration_idx = self.get_global_demonstration_index(demonstration_idx)
        val_demonstration_idx = self.get_global_demonstration_index(val_demonstration_idx)
        return demonstration_idx, val_demonstration_idx

    def get_global_demonstration_index(self, index):
        return int(
            sum(self.split[:self.dataset_type_idx]) * self.num_demonstrations
        ) + index


# TODO: validation data should be obtained from validation set, and actions applied as part of an inner evaluation loop
# TODO: within "step"
class DoubleDemonstrationIndexer(object):
    def __init__(self, demonstration_dataset, start, end, val_start, val_end):
        self.demonstration_dataset = demonstration_dataset
        self.start = start
        self.end = end
        self.val_start = val_start
        self.val_end = val_end

        self.indices = list(range(self.start, self.end + 1)) + list(range(self.val_start, self.val_end + 1))
        self.curr_idx = 0

    def advance(self):
        self.curr_idx += 1

    def get_curr_demonstration_data(self):
        return self.demonstration_dataset[self.get_curr_demonstration_idx()]

    def get_curr_demonstration_idx(self):
        assert self.curr_idx in range(len(self.indices))
        return self.indices[self.curr_idx]

    def training_demonstration_done(self):
        return self.curr_idx < len(self.indices) and self.get_curr_demonstration_idx() == self.val_start

    def validation_demonstration_done(self):
        return self.curr_idx == len(self.indices)

    def get_length(self):
        return len(self.indices)

    def get_training_length(self):
        return self.end - self.start + 1


# TODO: adapt code of CropDemonstration RL environments to use evaluator
class DemonstrationIndexer(object):
    def __init__(self, *start_end_pairs, demonstration_dataset):
        self.demonstration_dataset = demonstration_dataset
        self.start_end_pairs = start_end_pairs
        self.indices = reduce(lambda x, y: x + y,
                              list(map(lambda pair: list(range(pair[0], pair[1] + 1)), start_end_pairs)))
        self.curr_idx = 0

    def advance(self):
        self.curr_idx += 1

    def get_curr_demonstration_data(self):
        return self.demonstration_dataset[self.get_curr_demonstration_idx()]

    def get_curr_demonstration_idx(self):
        assert self.curr_idx in range(len(self.indices))
        return self.indices[self.curr_idx]

    def done(self):
        return self.curr_idx == len(self.indices)

    def get_length(self):
        return len(self.indices)


class CropDemonstrationEnv(ImageSpaceProviderEnv):
    def __init__(self, demonstration_dataset, config, random_provider=np.random.choice, estimator=TipVelocityEstimator,
                 dataset_type_idx=DatasetModality.TRAINING, skip_reward=False, init_from=None):
        super().__init__(config["size"])
        self.demonstration_dataset = demonstration_dataset
        self.config = config
        self.random_provider = random_provider
        # use_split_idx = 0 for training, 1 for validation, 2 for test
        self.demonstration_sampler = DemonstrationSampler(
            split=config["split"],
            num_demonstrations=self.demonstration_dataset.get_num_demonstrations(),
            dataset_type_idx=dataset_type_idx.value,
            random_provider=random_provider
        )

        self.init_from = init_from  # to use pretrained weights as initialisation
        self.parameter_state_dict = None
        if self.init_from is not None:
            self.parameter_state_dict = MetaImitationLearning.load_best_params(
                f"models/pretraining_test/{self.init_from}")
        self.estimator = estimator

        self.cropped_width, self.cropped_height = self.config["cropped_size"]
        self.width, self.height = self.config["size"]
        self.pixel_cropper = TrainingPixelROI(self.cropped_height, self.cropped_width, add_spatial_maps=True)
        self.to_tensor = torchvision.transforms.ToTensor()

        self.demonstration_states = []
        self.state = None
        self.demonstration_indexer = None
        # store the final training demonstration's crop to be able to apply it when computing the rewards
        self.final_training_crop = None

        self.epoch_list = []
        self.validation_list = []
        self.skip_reward = skip_reward  # skip reward computation when testing

    def reset(self):
        self.final_training_crop = None
        # sample new demonstration, one for training and one for validation, both DIFFERENT (replace = False)
        # but within correct split
        demonstration_idx, val_demonstration_idx = self.demonstration_sampler.sample_train_val_demonstrations()
        # training demonstration
        start, end = self.demonstration_dataset.get_indices_for_demonstration(demonstration_idx)
        # validation demonstration
        val_start, val_end = self.demonstration_dataset.get_indices_for_demonstration(val_demonstration_idx)
        self.demonstration_indexer = DoubleDemonstrationIndexer(
            self.demonstration_dataset, start, end, val_start, val_end
        )
        # state
        self.state = ImageOffsetState(self.demonstration_indexer.get_curr_demonstration_data())
        self.demonstration_states = [self.state]
        return self.state.get()

    def step(self, action):
        next_state, done = self.apply_action(action)
        center_crop_pixel = next_state.get_center_crop()
        if self.demonstration_indexer.training_demonstration_done():
            # we want to return to the agent the first validation state, with center crop, but want the reward function
            # to see the previous crop (reward func only uses list of demonstration_states)
            # next state has validation image and center crop
            self.demonstration_states.append(ImageOffsetState(next_state.get_data(), *self.final_training_crop))
            center_crop_pixel = self.final_training_crop
        else:
            # if done is True, still add to record last crop (dummy state with image = None)
            self.demonstration_states.append(next_state)

        reward = self.get_reward(done)
        self.state = next_state
        return self.state.get() if not done else self.dummy_observation, reward, done, dict(
            center_crop_pixel=center_crop_pixel)

    def apply_action(self, action):
        self.demonstration_indexer.advance()

        if self.demonstration_indexer.training_demonstration_done():
            # return starting state for validation demonstration
            # store final crop to apply to last training demonstration image
            self.final_training_crop = self.state.apply_action(
                None, action[0], action[1], self.cropped_width, self.cropped_height,
                restrict_crop_move=self.config["restrict_crop_move"]
            ).get_center_crop()
            return ImageOffsetState(self.demonstration_indexer.get_curr_demonstration_data()), False

        if self.demonstration_indexer.validation_demonstration_done():
            return self.state.apply_action(None, action[0], action[1], self.cropped_width, self.cropped_height,
                                           restrict_crop_move=self.config["restrict_crop_move"]), True

        new_state = self.state.apply_action(self.demonstration_indexer.get_curr_demonstration_data(), action[0],
                                            action[1],
                                            self.cropped_width, self.cropped_height,
                                            restrict_crop_move=self.config["restrict_crop_move"])
        return new_state, False

    def get_reward(self, done):
        if self.skip_reward:
            return 0

        if not done:
            return 0

        # all images are in self.demonstration_states, train with those
        # last demonstration state is dummy state to hold last crop
        assert len(self.demonstration_states) == self.demonstration_indexer.get_length() + 1

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
        training_dataset, validation_dataset = FromListsDataset(
            cropped_images, tip_velocities, rotations, keys=["image", "tip_velocities", "rotations"]
        ).split(self.demonstration_indexer.get_training_length())
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

        network_param = get_network_param_if_init_from(self.init_from, self.config, self.parameter_state_dict)

        estimator = self.estimator(
            batch_size=len(training_dataset),
            learning_rate=self.config["learning_rate"],
            image_size=self.config["cropped_size"],  # learning from cropped size
            **network_param,  # either initialise from class or from model
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
        self.validation_list.append(estimator.get_val_losses())
        return reward

    def get_epoch_list_stats(self):
        if len(self.epoch_list) == 0:
            raise ValueError("No estimators were trained")
        arr = np.array(self.epoch_list)
        return np.mean(arr), np.std(arr)

    def save_validation_losses_list(self, path):
        import torch
        torch.save(dict(validation_list=self.validation_list), path)


class FilterSpatialEvaluator(object):
    def __init__(self, test_env, num_iter=1):
        self.test_env = test_env
        self.num_iter = num_iter
        self.rl_model = None

    def set_rl_model(self, rl_model):
        self.rl_model = rl_model

    # returns top k selected features for every timestep
    def evaluate_and_get(self):
        if self.rl_model is None:
            raise ValueError("You forgot to set the RL model!")

        # episode is of length "num_iter" true episodes
        obs = self.test_env.reset(num_demonstrations=self.num_iter)
        done = False
        while not done:
            action, _states = self.rl_model.predict(obs, deterministic=True)
            obs, _, done, info = self.test_env.step(action)

        return self.test_env.get_selected_features(), self.test_env.get_target_predictions()


class FilterSpatialFeatureSpaceProvider(gym.Env, ABC):
    def __init__(self, latent_dimension):
        super().__init__()
        # one action per spatial feature
        num_spatial_features = latent_dimension // 2
        # represent importance of spatial feature as value in -1.0 to 1.0 -> then need to rescale to range 0, 1
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_spatial_features,))
        # observation are spatial features, so vector of size latent dimension, should also be normalised in -1.0, 1.0
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(latent_dimension,))

    def render(self, mode='human'):
        raise NotImplementedError("Environment is not renderable... yet :(")
        pass


class FilterSpatialFeatureEnv(FilterSpatialFeatureSpaceProvider):

    def __init__(self, latent_dimension, feature_provider, demonstration_dataset, split, dataset_type_idx, device,
                 evaluator=None, num_average_training=3, k=10, random_provider=np.random.choice, skip_reward=False):
        super().__init__(latent_dimension)
        self.feature_provider = feature_provider
        self.latent_dimension = latent_dimension
        self.k = k
        self.num_average_training = num_average_training
        self.features = None
        self.target_predictions = None
        self.demonstration_dataset = demonstration_dataset
        self.demonstration_sampler = DemonstrationSampler(
            split=split,
            dataset_type_idx=dataset_type_idx.value,
            num_demonstrations=self.demonstration_dataset.get_num_demonstrations(),
            random_provider=random_provider
        )

        self.demonstration_indexer = None
        self.state = None
        self.device = device

        self.pixels = None

        self.evaluator = evaluator  # evaluator may be None for test environments
        self.skip_reward = skip_reward
        if self.evaluator is not None and self.skip_reward:
            raise ValueError("We need an evaluator to compute the reward on a validation set!")

    def get_selected_features(self):
        # (episode length, k * 2)
        return self.features

    def get_target_predictions(self):
        return self.target_predictions

    def get_np_pixels(self):
        # (episode length, 2)
        res = np.array(list(map(lambda p_trch: p_trch.numpy(), self.pixels)))
        return res

    def set_rl_model(self, rl_model):
        if self.evaluator is not None:
            self.evaluator.set_rl_model(rl_model)

    def step(self, action):
        self.demonstration_indexer.advance()
        # get scores from action
        top_k_features = self.state.get_top_k_features(action)
        assert top_k_features.shape == (self.k * 2,)
        self.features.append(top_k_features)

        if not self.demonstration_indexer.done():
            data = self.demonstration_indexer.get_curr_demonstration_data()
            self.target_predictions.append(data["target_vel_rot"])
            spatial_features = self.feature_provider(data["images"][1]).view(self.latent_dimension).cpu().numpy()
            self.state = FilterSpatialFeatureState(self.k, spatial_features=spatial_features)
            self.pixels.append(data["pixel"])
            return spatial_features, 0, False, {}

        if self.skip_reward:
            # for test environments
            return None, -1, True, {}

        # if last demonstration,instead, train, return neg val loss, set done to true
        # get validation dataset features and target predictions
        val_features, val_target_predictions = self.evaluator.evaluate_and_get()
        training_dataset, validation_dataset = FromListsDataset(
            self.features + val_features, self.target_predictions + val_target_predictions,
            keys=["features", "target_vel_rot"]
        ).split(self.demonstration_indexer.get_length())
        train_dataloader = DataLoader(training_dataset, batch_size=len(training_dataset), shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=True)

        losses = []
        # average loss
        for _ in range(self.num_average_training):
            action_predictor = DSAE_TargetActionPredictor(k=self.k)
            optimiser = torch.optim.Adam(action_predictor.parameters(), lr=0.001)
            action_predictor_manager = ActionPredictorManager(
                action_predictor=action_predictor,
                num_epochs=100,
                optimiser=optimiser,
                device=self.device
            )

            action_predictor_manager.train(train_dataloader, validation_dataloader)
            losses.append(action_predictor_manager.get_validation_loss())

        reward = - np.mean(losses)
        return np.ones(self.latent_dimension), reward, True, {}

    def reset(self, num_demonstrations=1):
        # num_demonstrations: number of training demonstrations to sample
        # when we have a test environment that draws demonstrations from the validation set, this is useful to we
        # sample without replacement a number of demonstrations for validation, rather than just one
        # this is used by the evaluator
        self.demonstration_indexer = self.demonstration_sampler.get_demonstration_indexer(
            demonstration_dataset=self.demonstration_dataset, demonstrations=num_demonstrations
        )
        data = self.demonstration_indexer.get_curr_demonstration_data()
        spatial_features = self.feature_provider(data["images"][1]).view(self.latent_dimension).cpu().numpy()
        self.features = []
        self.target_predictions = [data["target_vel_rot"]]
        self.state = FilterSpatialFeatureState(k=self.k, spatial_features=spatial_features)
        self.pixels = [data["pixel"]]
        return spatial_features
