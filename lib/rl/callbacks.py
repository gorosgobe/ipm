import numpy as np
import tensorflow as tf
from stable_baselines.common.callbacks import BaseCallback

from lib.rl.demonstration_env import CropDemonstrationEnv
from lib.rl.utils import CropTester
from lib.common.test_utils import get_distances_between_chosen_features_and_pixels


class FeatureDistanceScoreCallback(BaseCallback):
    def __init__(self, test_env, n_episodes=5, every=10000, verbose=0):
        super().__init__(verbose)
        self.test_env = test_env
        self.n_episodes = n_episodes
        self.every = every

    def _on_step(self):
        if self.num_timesteps > 0 and self.num_timesteps % self.every == 0:
            obs = self.test_env.reset(num_demonstrations=self.n_episodes)
            done = False
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, _, done, info = self.test_env.step(action)

            # (length episode, k * 2)
            features = np.array(self.test_env.get_selected_features())
            # (length episode, 2)
            pixels = self.test_env.get_np_pixels()
            norms = get_distances_between_chosen_features_and_pixels(
                features=features, pixels=pixels
            )
            mean = np.mean(norms)
            summary_mean = tf.Summary(
                value=[tf.Summary.Value(tag=f"feature_distance_mean", simple_value=mean)])
            self.locals['writer'].add_summary(summary_mean, self.num_timesteps)

            minimum = np.mean(np.min(norms, axis=-1))
            summary_minimum = tf.Summary(
                value=[tf.Summary.Value(tag=f"feature_distance_minimum", simple_value=minimum)])
            self.locals['writer'].add_summary(summary_minimum, self.num_timesteps)

            #TODO: extend for when training with more than one demonstration (?, this means more sparsity, so maybe not?)

            # mean at the beginning of the demonstration - intuitively (?) this should be close to target, especially for
            # np.min case as opposed to np.mean
            # norms is (episode length, k)
            initial_mean = np.mean(np.min(norms[:3, :], axis=-1))
            summary_initial_mean = tf.Summary(
                value=[tf.Summary.Value(tag=f"feature_distance_initial_mean", simple_value=initial_mean)])
            self.locals['writer'].add_summary(summary_initial_mean, self.num_timesteps)

            initial_minimum = np.mean(norms[:3, :])
            summary_initial_minimum = tf.Summary(
                value=[tf.Summary.Value(tag=f"feature_distance_initial_mean", simple_value=initial_minimum)])
            self.locals['writer'].add_summary(summary_initial_minimum, self.num_timesteps)

        return True


class CropScoreCallback(BaseCallback):
    """
    Logs a produced score to Tensorboard.
    """

    def __init__(self, score_name, score_function, prefix, config, demonstration_dataset, crop_test_modality,
                 compute_score_every, log_dir, number_rollouts, save_images_every=-1, verbose=0,
                 environment_klass=CropDemonstrationEnv):
        super().__init__(verbose)
        self.score_name = score_name
        self.score_function = score_function
        self.crop_tester = CropTester(config, demonstration_dataset, crop_test_modality,
                                      environment_klass=environment_klass)
        self.log_dir = log_dir
        self.prefix = prefix  # when saving images
        self.compute_score_every = compute_score_every
        self.number_rollouts = number_rollouts  # number of times to sample environment
        # TODO: change to use indexer, and reset with num_demonstrations, that should do it
        self.random = True  # Sample randomly from the environment?
        self.save_images_every = save_images_every  # if -1, dont save images
        self.count_rollouts = 0

    def _on_rollout_start(self):
        self.count_rollouts += 1

        if self.count_rollouts % self.compute_score_every != 0:
            return

        value_means, value_stds = [], []
        for i in range(self.number_rollouts):
            value_mean, value_std = self.crop_tester.get_crop_score_per_rollout(
                criterion=self.score_function,
                model=self.model,
                # save images only for one rollout
                save_images=self.save_images_every != -1 and self.count_rollouts % self.save_images_every == 0 and i == 0,
                log_dir=self.log_dir,
                prefix=f"{self.count_rollouts}_{self.prefix}"
            )
            value_means.append(value_mean)
            value_stds.append(value_std)

        # mean +- std (of means)
        value_means = np.array(value_means)
        value_mean_overall = np.mean(value_means)
        value_mean_std_overall = np.std(value_means)

        # mean +- std
        summary_mean = tf.Summary(
            value=[tf.Summary.Value(tag=f"{self.score_name}_mean", simple_value=value_mean_overall)])
        self.locals['writer'].add_summary(summary_mean, self.num_timesteps)
        summary_mean_std = tf.Summary(
            value=[tf.Summary.Value(tag=f"{self.score_name}_mean_std", simple_value=value_mean_std_overall)])
        self.locals['writer'].add_summary(summary_mean_std, self.num_timesteps)

        # average std
        value_std_overall = np.mean(np.array(value_stds))
        summary_std = tf.Summary(value=[tf.Summary.Value(tag=f"{self.score_name}_std", simple_value=value_std_overall)])
        self.locals['writer'].add_summary(summary_std, self.num_timesteps)
