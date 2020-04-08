import tensorflow as tf
import numpy as np

from stable_baselines.common.callbacks import BaseCallback

from lib.rl.demonstration_env import SingleDemonstrationEnv
from lib.rl.utils import CropTester


class ScoreCallback(BaseCallback):
    """
    Logs a produced score to Tensorboard.
    """

    def __init__(self, score_name, score_function, prefix, config, demonstration_dataset, crop_test_modality,
                 compute_score_every, log_dir, number_rollouts, save_images_every=-1, verbose=0, environment_klass=SingleDemonstrationEnv):
        super().__init__(verbose)
        self.score_name = score_name
        self.score_function = score_function
        self.crop_tester = CropTester(config, demonstration_dataset, crop_test_modality, environment_klass=environment_klass)
        self.log_dir = log_dir
        self.prefix = prefix  # when saving images
        self.compute_score_every = compute_score_every
        self.number_rollouts = number_rollouts  # number of times to sample environment
        # TODO: implement
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
        summary_mean = tf.Summary(value=[tf.Summary.Value(tag=f"{self.score_name}_mean", simple_value=value_mean_overall)])
        self.locals['writer'].add_summary(summary_mean, self.num_timesteps)
        summary_mean_std = tf.Summary(value=[tf.Summary.Value(tag=f"{self.score_name}_mean_std", simple_value=value_mean_std_overall)])
        self.locals['writer'].add_summary(summary_mean_std, self.num_timesteps)

        # average std
        value_std_overall = np.mean(np.array(value_stds))
        summary_std = tf.Summary(value=[tf.Summary.Value(tag=f"{self.score_name}_std", simple_value=value_std_overall)])
        self.locals['writer'].add_summary(summary_std, self.num_timesteps)


class SaveCropCallback(BaseCallback):
    """
    Samples an environment episode and saves the images, drawing the ground truth, expected crop in green and
    the predicted crop by the RL agent in red.
    """

    def __init(self, save_every, log_dir, verbose=0):
        super().__init__(verbose)
        self.save_every = save_every
        self.log_dir = log_dir
        self.count = 0

    def _on_rollout_start(self):
        if self.count % self.save_every:
            pass
        self.count += 1
