from abc import ABC

import gym
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from lib.cv.controller import TrainingPixelROI
from lib.cv.dataset import FromListsDataset
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.rl.demonstration_ctrl import DemonstrationSampler
from lib.rl.demonstration_env import CropDemonstrationUtils
from lib.rl.state import ImageOffsetState, SpatialOffsetState


class SpatialFeatureCropSpaceProvider(gym.Env, ABC):
    def __init__(self, latent_dimension, scale=None):
        super().__init__()
        # represent offset to move the crop by
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        # observation are spatial features + center of previous crop, all in range [-1.0, 1.0]
        # + scale of crop, if scale flag is set
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(latent_dimension + 2 + (scale is not None),))

    def render(self, mode='human'):
        raise NotImplementedError("Not going to render this environment")
        pass


class SpatialFeatureCropEnv(SpatialFeatureCropSpaceProvider):
    def __init__(self, latent_dimension, demonstration_dataset, split, dataset_type_idx, cropped_size, device,
                 network_klass, num_training_demonstrations=None, evaluator=None, random_provider=np.random.choice,
                 skip_reward=False, sparse=True, restrict_crop_move=None, shuffle=True, estimator=TipVelocityEstimator,
                 scale=False, decrease_scale_every=int(1e5), size=(128, 96)):
        if scale:
            scale = 1.0
            print(f"Environment with scale, decrease at rate 1/{decrease_scale_every} steps")
        else:
            scale = None
        super().__init__(latent_dimension, scale=scale)
        self.scale = scale
        self.decrease_scale_every = decrease_scale_every
        self.steps = 0
        self.width, self.height = size
        self.cropped_width, self.cropped_height = cropped_size
        self.to_tensor = torchvision.transforms.ToTensor()
        self.latent_dimension = latent_dimension
        self.network_klass = network_klass
        self.num_training_demonstrations = num_training_demonstrations
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

    def get_pixel_cropper(self):
        if self.scale is None:
            pixel_cropper = TrainingPixelROI(self.cropped_height, self.cropped_width, add_spatial_maps=True)
        else:
            scale = (self.scale + 1) / 2
            assert 0.0 <= scale <= 1.0
            pixel_cropper = TrainingPixelROI(
                cropped_height=int(self.cropped_height + (self.height - self.cropped_height) * scale),
                cropped_width=int(self.cropped_width + (self.width - self.cropped_width) * scale),
                add_spatial_maps=True
            )
        return pixel_cropper

    def decrease_crop(self):
        # updates scale, decreases by one pixel the shortest dimension of the image
        # solve a + (b - a) * s' = a + (b - a) * s - 1
        # assumes crop has same image ratio as original size
        scale = (self.scale + 1) / 2
        b = min(self.width, self.height)
        a = min(self.cropped_width, self.cropped_height)
        new_scale = scale - (1 / (b - a))
        self.scale = 2 * max(new_scale, 0) - 1

    def get_curr_demonstration_idx(self):
        # this is called after step() by callbacks, so demonstration index has advanced and we returned the pixel information
        # of the previous image - hence the previous index is required
        return self.demonstration_indexer.get_prev_demonstration_idx()

    def get_scale(self):
        return self.scale

    def reset(self, num_demonstrations=1):
        if self.num_training_demonstrations is not None:
            # training data has num_training_demonstrations true episodes
            num_demonstrations = self.num_training_demonstrations

        self.steps += 1
        if self.scale is not None and self.steps % self.decrease_scale_every == 0:
            self.decrease_crop()
            print("Scale: ", self.scale)

        self.demonstration_indexer = self.demonstration_sampler.get_demonstration_indexer(
            demonstration_dataset=self.demonstration_dataset, demonstrations=num_demonstrations
        )

        self.states = []

        data = self.demonstration_indexer.get_curr_demonstration_data()
        spatial_features = data["features"].cpu().numpy()
        image_offset_state = ImageOffsetState(data=data)
        self.state = SpatialOffsetState(
            spatial_features=spatial_features,
            image_offset_state=image_offset_state,
            scale=self.scale
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
                restrict_crop_move=self.restrict_crop_move, scale=self.scale
            )
            center_crop_pixel = self.state.get_center_crop()
            return self.state.get(), 0, False, dict(center_crop_pixel=center_crop_pixel)

        # at the end, so add dummy state to hold last crop
        self.states.append(
            self.state.apply_action(
                spatial_features=None, data=None, dx=action[0], dy=action[1],
                cropped_width=self.cropped_width, cropped_height=self.cropped_height,
                restrict_crop_move=self.restrict_crop_move, scale=self.scale
            )
        )
        assert len(self.states) == self.demonstration_indexer.get_length() + 1

        if self.skip_reward:
            return None, -1, True, dict(center_crop_pixel=self.states[-1].get_center_crop())

        # train network to get reward
        cropped_training_images, training_tip_velocities, training_rotations = self.get_processed_data_from_states()
        assert len(cropped_training_images) == len(training_tip_velocities) == len(training_rotations)
        cropped_validation_images, validation_tip_velocities, validation_rotations = self.evaluator.evaluate_and_get(
            scale=self.scale
        )
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

        train_data_loader = DataLoader(
            training_dataset,
            batch_size=min(len(training_dataset), 32),
            num_workers=0,  # in memory, so none needed
            shuffle=self.shuffle
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=min(len(validation_dataset), 32),
            num_workers=0,
            shuffle=self.shuffle
        )

        image_size = (self.cropped_width, self.cropped_height)
        network_klass = self.network_klass
        if self.scale is not None:
            scale = (self.scale + 1) / 2
            image_size = (
                int(self.cropped_width + (self.width - self.cropped_width) * scale),
                int(self.cropped_height + (self.height - self.cropped_height) * scale)
            )
            # network class is general, need to create one for the specific type
            network_klass = network_klass.create(*image_size)

        estimator = self.estimator(
            batch_size=len(training_dataset),
            learning_rate=0.0001,
            image_size=image_size,
            network_klass=network_klass,
            device=self.device,
            patience=10,
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
        return np.ones(self.latent_dimension + 2 + (self.scale is not None)), reward, True, dict(
            center_crop_pixel=self.states[-1].get_center_crop())

    def get_processed_data_from_states(self):
        # get pixel cropper, using current scale
        pixel_cropper = self.get_pixel_cropper()
        return CropDemonstrationUtils.get_processed_crop_info(
            states_with_images_and_crops=self.states,
            pixel_cropper=pixel_cropper,
            to_tensor=self.to_tensor
        )
