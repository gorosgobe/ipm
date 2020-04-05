import sys
import warnings

from stable_baselines.common.policies import ActorCriticPolicy
import numpy as np
import tensorflow as tf
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc


class NoScaleCNNFeedForwardPolicy(ActorCriticPolicy):
    """
    Adapted from stable_baselines.common.policies.FeedForwardPolicy
    """
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, cnn_extractor, reuse=False, **kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)

        with tf.variable_scope("model", reuse=reuse):
            pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})


class PPOPolicy(NoScaleCNNFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                         cnn_extractor=self.extractor, **kwargs)

    @staticmethod
    def extractor(observations, image_size, **kwargs):
        width, height = image_size
        observation_size = observations.shape[1]
        center_previous, images_1d = tf.split(observations, axis=1, num_or_size_splits=[2, int(observation_size - 2)])
        images = tf.reshape(images_1d, (-1, height, width, 3))
        tf.print(images, output_stream=sys.stderr)
        activ = tf.nn.relu
        out_conv1 = activ(conv(images, "c1", n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
        out_conv2 = activ(conv(out_conv1, "c2", n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
        out_conv3 = activ(conv(out_conv2, "c3", n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
        out_conv3_flattened = conv_to_fc(out_conv3)
        out_fc1 = activ(linear(out_conv3_flattened, "fc1", n_hidden=126, init_scale=np.sqrt(2)))
        concatenated = tf.concat(axis=1, values=[out_fc1, center_previous])
        return activ(linear(concatenated, "fc2", n_hidden=64, init_scale=np.sqrt(2)))


