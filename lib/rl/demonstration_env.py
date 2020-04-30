from abc import ABC

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
from lib.rl.demonstration_ctrl import DemonstrationSampler
from lib.rl.state import ImageOffsetState, FilterSpatialFeatureState, SpatialOffsetState
from lib.rl.utils import DatasetModality, CropScorer


class CropDemonstrationUtils(object):
    @staticmethod
    def get_processed_crop_info(states_with_images_and_crops, pixel_cropper, to_tensor):
        cropped_images_and_bounding_boxes = (
            pixel_cropper.crop(
                states_with_images_and_crops[i].get_np_image(),
                states_with_images_and_crops[i + 1].get_center_crop()
            ) for i in range(len(states_with_images_and_crops) - 1)
        )

        cropped_images = map(lambda img_n_box: img_n_box[0], cropped_images_and_bounding_boxes)
        cropped_images = list(map(lambda img: to_tensor(img), cropped_images))
        tip_velocities = [state.get_tip_velocity() for state in states_with_images_and_crops[:-1]]
        rotations = [state.get_rotations() for state in states_with_images_and_crops[:-1]]
        return cropped_images, tip_velocities, rotations


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


class CropDemonstrationEnv(ImageSpaceProviderEnv):
    def __init__(self, demonstration_dataset, config, random_provider=np.random.choice, estimator=TipVelocityEstimator,
                 dataset_type_idx=DatasetModality.TRAINING, skip_reward=False, init_from=None, evaluator=None,
                 sparse=True):
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

        # should the reward be sparse or dense? If dense, we train an NN at every env step
        self.sparse = sparse
        self.evaluator = evaluator  # evaluator may be None for test environments
        self.skip_reward = skip_reward  # skip reward computation when testing

        if self.evaluator is None and not self.skip_reward:
            raise ValueError("We need an evaluator to compute the reward on a validation set!")

    def reset(self, num_demonstrations=1):
        self.final_training_crop = None

        self.demonstration_indexer = self.demonstration_sampler.get_demonstration_indexer(
            demonstration_dataset=self.demonstration_dataset, demonstrations=num_demonstrations
        )
        # state
        self.state = ImageOffsetState(self.demonstration_indexer.get_curr_demonstration_data())
        self.demonstration_states = [self.state]
        return self.state.get()

    def step(self, action):
        next_state, done = self.apply_action(action)
        center_crop_pixel = next_state.get_center_crop()
        self.demonstration_states.append(next_state)
        reward = self.get_reward(done)
        self.state = next_state
        return self.state.get() if not done else self.dummy_observation, reward, done, dict(
            center_crop_pixel=center_crop_pixel
        )

    def apply_action(self, action):
        self.demonstration_indexer.advance()

        if self.demonstration_indexer.done():
            self.final_training_crop = self.state.apply_action(
                None, action[0], action[1], self.cropped_width, self.cropped_height,
                restrict_crop_move=self.config["restrict_crop_move"]
            )
            return self.final_training_crop, True

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

        cropped_training_images, training_tip_velocities, training_rotations = self.get_processed_data_from_states()
        cropped_validation_images, validation_tip_velocities, validation_rotations = self.evaluator.evaluate_and_get()

        # Use first demonstration for training, second for validation (correlations within demonstrations)
        training_dataset, validation_dataset = FromListsDataset(
            cropped_training_images + cropped_validation_images,
            training_tip_velocities + validation_tip_velocities,
            training_rotations + validation_rotations,
            keys=["image", "tip_velocities", "rotations"]
        ).split(self.demonstration_indexer.get_length())

        # train with everything as the batch, its small anyways
        train_data_loader = DataLoader(
            training_dataset,
            batch_size=len(training_dataset),
            num_workers=0,  # in memory, so none needed
            shuffle=self.config["shuffle"]
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=len(validation_dataset),
            num_workers=0,
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

    def get_processed_data_from_states(self):
        return CropDemonstrationUtils.get_processed_crop_info(
            states_with_images_and_crops=self.demonstration_states,
            pixel_cropper=self.pixel_cropper,
            to_tensor=self.to_tensor
        )

    def get_epoch_list_stats(self):
        if len(self.epoch_list) == 0:
            raise ValueError("No estimators were trained")
        arr = np.array(self.epoch_list)
        return np.mean(arr), np.std(arr)

    def save_validation_losses_list(self, path):
        import torch
        torch.save(dict(validation_list=self.validation_list), path)


class SpatialFeatureCropSpaceProvider(gym.Env, ABC):
    def __init__(self, latent_dimension):
        super().__init__()
        # represent offset to move the crop by
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        # observation are spatial features + center of previous crop, all in range [-1.0, 1.0]
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(latent_dimension + 2,))

    def render(self, mode='human'):
        raise NotImplementedError("Not going to render this environment")
        pass


class SpatialFeatureCropEnv(SpatialFeatureCropSpaceProvider):
    def __init__(self, latent_dimension, demonstration_dataset, split, dataset_type_idx, cropped_size, device,
                 network_klass, evaluator=None, random_provider=np.random.choice, skip_reward=False, sparse=True,
                 restrict_crop_move=None, shuffle=True, estimator=TipVelocityEstimator):
        super().__init__(latent_dimension)
        self.cropped_width, self.cropped_height = cropped_size
        self.pixel_cropper = TrainingPixelROI(self.cropped_height, self.cropped_width, add_spatial_maps=True)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.latent_dimension = latent_dimension
        self.network_klass = network_klass
        self.estimator = estimator
        # do we want a sparse or dense reward signal? If dense, we train an NN at every env step
        self.sparse = sparse
        self.demonstration_dataset = demonstration_dataset
        self.demonstration_sampler = DemonstrationSampler(
            split=split,
            dataset_type_idx=dataset_type_idx.value,
            num_demonstrations=self.demonstration_dataset.get_num_demonstrations(),
            random_provider=random_provider
        )

        self.demonstration_indexer = None
        self.states = None
        self.state = None
        self.device = device

        self.restrict_crop_move = restrict_crop_move
        self.shuffle = shuffle
        self.evaluator = evaluator  # evaluator may be None for test environments
        self.skip_reward = skip_reward
        if self.evaluator is None and not self.skip_reward:
            raise ValueError("We need an evaluator to compute the reward on a validation set!")

    def reset(self, num_demonstrations=1):
        self.demonstration_indexer = self.demonstration_sampler.get_demonstration_indexer(
            demonstration_dataset=self.demonstration_dataset, demonstrations=num_demonstrations
        )

        self.states = []

        data = self.demonstration_indexer.get_curr_demonstration_data()
        spatial_features = data["features"].cpu().numpy()
        image_offset_state = ImageOffsetState(data=data)
        self.state = SpatialOffsetState(
            spatial_features=spatial_features,
            image_offset_state=image_offset_state
        )
        return self.state.get()

    def step(self, action):
        self.states.append(self.state)
        self.demonstration_indexer.advance()

        if not self.demonstration_indexer.done():
            data = self.demonstration_indexer.get_curr_demonstration_data()
            spatial_features = data["features"].cpu().numpy()
            self.state = self.state.apply_action(
                spatial_features=spatial_features, data=data, dx=action[0], dy=action[1],
                cropped_width=self.cropped_width, cropped_height=self.cropped_height,
                restrict_crop_move=self.restrict_crop_move
            )
            center_crop_pixel = self.state.get_center_crop()
            return self.state.get(), 0, False, dict(center_crop_pixel=center_crop_pixel)

        if self.skip_reward:
            return None, -1, True, {}

        # at the end, so add dummy state to hold last crop
        self.states.append(
            self.state.apply_action(
                spatial_features=None, data=None, dx=action[0], dy=action[1],
                cropped_width=self.cropped_width, cropped_height=self.cropped_height,
                restrict_crop_move=self.restrict_crop_move
            )
        )
        assert len(self.states) == self.demonstration_indexer.get_length() + 1
        # train network to get reward
        cropped_training_images, training_tip_velocities, training_rotations = self.get_processed_data_from_states()
        assert len(cropped_training_images) == len(training_tip_velocities) == len(training_rotations)
        cropped_validation_images, validation_tip_velocities, validation_rotations = self.evaluator.evaluate_and_get()
        assert len(cropped_validation_images) == len(validation_tip_velocities) == len(validation_rotations)

        # Use first demonstration for training, second for validation (correlations within demonstrations)
        training_dataset, validation_dataset = FromListsDataset(
            cropped_training_images + cropped_validation_images,
            training_tip_velocities + validation_tip_velocities,
            training_rotations + validation_rotations,
            keys=["image", "tip_velocities", "rotations"]
        ).split(self.demonstration_indexer.get_length())

        assert len(training_dataset) == len(cropped_training_images)
        assert len(validation_dataset) == len(cropped_validation_images)

        # train with everything as the batch, its small anyways
        train_data_loader = DataLoader(
            training_dataset,
            batch_size=len(training_dataset),
            num_workers=0,  # in memory, so none needed
            shuffle=self.shuffle
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=len(validation_dataset),
            num_workers=0,
            shuffle=self.shuffle
        )

        estimator = self.estimator(
            batch_size=len(training_dataset),
            learning_rate=0.0001,
            image_size=(self.cropped_width, self.cropped_height),
            network_klass=self.network_klass,
            device=self.device,
            patience=100,
            verbose=False
        )

        estimator.train(
            data_loader=train_data_loader,
            max_epochs=100,
            validate_epochs=1,
            val_loader=validation_data_loader,
        )
        reward = -estimator.get_best_val_loss()

        # dummy observation, needed for SAC
        return np.ones(self.latent_dimension), reward, True, dict(center_crop_pixel=self.states[-1].get_center_crop())

    def get_processed_data_from_states(self):
        return CropDemonstrationUtils.get_processed_crop_info(
            states_with_images_and_crops=self.states,
            pixel_cropper=self.pixel_cropper,
            to_tensor=self.to_tensor
        )


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
    def __init__(self, latent_dimension, demonstration_dataset, split, dataset_type_idx, device,
                 evaluator=None, num_training_demonstrations=None, num_average_training=3, k=10,
                 random_provider=np.random.choice, skip_reward=False, sparse=True):
        super().__init__(latent_dimension)
        self.latent_dimension = latent_dimension
        self.k = k
        self.num_average_training = num_average_training
        self.num_training_demonstrations = num_training_demonstrations
        self.features = None
        # do we want a sparse or dense reward signal? If dense, we train an NN at every env step
        self.sparse = sparse
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
        if self.evaluator is None and not self.skip_reward:
            raise ValueError("We need an evaluator to compute the reward on a validation set!")

    def get_selected_features(self):
        # (episode length, k * 2)
        return self.features

    def get_target_predictions(self):
        return self.target_predictions

    def get_episode_demonstration_lengths(self):
        # an episode may be composed of multiple demonstrations, so return the length of each training demonstration
        # involved in the episode
        assert self.demonstration_indexer is not None
        return self.demonstration_indexer.get_lengths()

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
        prev_data = self.demonstration_indexer.get_prev_demonstration_data()
        self.target_predictions.append(prev_data["target_vel_rot"])
        self.pixels.append(prev_data["pixel"])

        # dummy value in case the episode is done
        spatial_features = np.ones(self.latent_dimension)
        if not self.demonstration_indexer.done():
            # set next state
            data = self.demonstration_indexer.get_curr_demonstration_data()
            spatial_features = data["features"].cpu().numpy()
            self.state = FilterSpatialFeatureState(self.k, spatial_features=spatial_features)
            if self.sparse:
                # shortcircuit
                return spatial_features, 0, False, {}

        if self.skip_reward:
            # for test environments
            return None, -1, True, {}

        # whether sparse or dense reward, we still train the network in the same way

        assert len(self.features) == len(self.target_predictions)
        num_training = len(self.features)
        # get validation dataset features and target predictions
        val_features, val_target_predictions = self.evaluator.evaluate_and_get()
        assert len(val_features) == len(val_target_predictions)

        training_dataset, validation_dataset = FromListsDataset(
            self.features + val_features, self.target_predictions + val_target_predictions,
            keys=["features", "target_vel_rot"]
        ).split(num_training)
        assert len(training_dataset) == num_training

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
        return spatial_features, reward, self.demonstration_indexer.done(), {}

    def reset(self, num_demonstrations=1):
        # num_demonstrations: number of training demonstrations to sample
        # when we have a test environment that draws demonstrations from the validation set, this is useful so we
        # sample without replacement a number of demonstrations for validation, rather than just one
        # this is used by the evaluator
        if self.num_training_demonstrations is not None:
            # training data has num_training_demonstrations true episodes
            num_demonstrations = self.num_training_demonstrations

        self.demonstration_indexer = self.demonstration_sampler.get_demonstration_indexer(
            demonstration_dataset=self.demonstration_dataset, demonstrations=num_demonstrations
        )
        data = self.demonstration_indexer.get_curr_demonstration_data()
        spatial_features = data["features"].cpu().numpy()
        self.features = []
        self.target_predictions = []
        self.state = FilterSpatialFeatureState(k=self.k, spatial_features=spatial_features)
        self.pixels = []
        return spatial_features
