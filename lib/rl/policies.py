import numpy as np
import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, mlp
from stable_baselines.sac.policies import SACPolicy, LOG_STD_MIN, LOG_STD_MAX, gaussian_likelihood, gaussian_entropy, \
    apply_squashing_func


def extractor(observations, image_size, crop_size=(32, 24), add_coord=False, tile=False, **kwargs):
    width, height = image_size
    observation_size = observations.shape[1]
    center_previous, images_1d = tf.split(observations, axis=1, num_or_size_splits=[2, int(observation_size - 2)])
    images = tf.reshape(images_1d, (-1, height, width, 3))
    if add_coord:
        images = add_coord_channels(images, image_size)
    activ = tf.nn.relu
    out_conv1 = activ(conv(images, "c1", n_filters=64, filter_size=5, stride=2, pad="SAME", init_scale=np.sqrt(2), **kwargs))

    if not tile:
        out_conv2 = activ(conv(out_conv1, "c2", n_filters=64, filter_size=5, stride=2, pad="SAME", init_scale=np.sqrt(2), **kwargs))
        out_conv3 = activ(conv(out_conv2, "c3", n_filters=64, filter_size=5, stride=2, pad="SAME", init_scale=np.sqrt(2), **kwargs))
        out_conv3_flattened = conv_to_fc(out_conv3)
        out_fc1 = activ(linear(out_conv3_flattened, "fc1", n_hidden=62, init_scale=np.sqrt(2)))
        # normalise centers to -1, 1
        center_previous_normalised = (center_previous / tf.constant([width - 1, height - 1], dtype=tf.float32)) * 2 - 1
        out = tf.concat(axis=1, values=[out_fc1, center_previous_normalised])
    else:
        out_conv2 = activ(conv(out_conv1, "c2", n_filters=62, filter_size=5, stride=2, pad="SAME", init_scale=np.sqrt(2), **kwargs))
        conv2_shape = tf.shape(out_conv2)

        tl, br = get_tl_br(center_previous, *crop_size)
        # normalise tl, br to -1, 1
        tl = (tl / tf.constant([width - 1, height - 1], dtype=tf.float32)) * 2 - 1
        br = (tl / tf.constant([width - 1, height - 1], dtype=tf.float32)) * 2 - 1

        tl_tiled = get_tile_map(tl, conv2_shape[1], conv2_shape[2])
        br_tiled = get_tile_map(br, conv2_shape[1], conv2_shape[2])
        # concatenate tile maps for center_previous_normalised
        out_conv2_tile = tf.concat(axis=-1, values=[out_conv2, tl_tiled, br_tiled])
        out_conv3 = activ(conv(out_conv2_tile, "c3", n_filters=64, filter_size=5, stride=2, pad="SAME", init_scale=np.sqrt(2), **kwargs))
        out_conv3_flattened = conv_to_fc(out_conv3)
        out = activ(linear(out_conv3_flattened, "fc1", n_hidden=64, init_scale=np.sqrt(2)))

    return out


def get_tl_br(centers, cropped_width, cropped_height):
    tl = centers - tf.constant([cropped_width // 2, cropped_height // 2], dtype=tf.float32)
    br = centers + tf.constant([cropped_width // 2, cropped_height // 2], dtype=tf.float32)

    if cropped_width % 2 == 0:
        tl = tl + tf.constant([1, 0], dtype=tf.float32)

    if cropped_height % 2 == 0:
        tl = tl + tf.constant([0, 1], dtype=tf.float32)

    return tl, br


def get_tile_map(corner, h, w):
    w_tiled = tf.tile(corner, [1, w // 2])
    w_tiled_expanded = tf.expand_dims(w_tiled, 1)
    w_h_tiled = tf.tile(w_tiled_expanded, [1, h, 1])
    out = tf.expand_dims(w_h_tiled, -1)
    return out


def add_coord_channels(image_batch, image_size):
    # Adapted from code provided in https://arxiv.org/pdf/1807.03247.pdf
    # An Intriguing Failing of convolutional neural networks and the CoordConv solution, Liu et al., 2018
    width, height = image_size
    batch_size_tensor = tf.shape(image_batch)[0]
    x_dim = height
    y_dim = width
    # (batch size, x_dim, y_dim, channels) = (b, h, w, c)
    xx_ones = tf.ones([batch_size_tensor, x_dim], dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, -1)
    xx_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0), [batch_size_tensor, 1])
    xx_range = tf.expand_dims(xx_range, 1)
    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)
    yy_ones = tf.ones([batch_size_tensor, y_dim], dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, 1)
    yy_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0), [batch_size_tensor, 1])

    yy_range = tf.expand_dims(yy_range, -1)
    yy_channel = tf.matmul(yy_range, yy_ones)
    yy_channel = tf.expand_dims(yy_channel, -1)
    xx_channel = tf.cast(xx_channel, "float32") / (x_dim - 1)
    yy_channel = tf.cast(yy_channel, "float32") / (y_dim - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    # first i channel, then j channel, like in the rest of our code
    return tf.concat([image_batch, yy_channel, xx_channel], axis=-1)


class PPONoScaleCNNFeedForwardPolicy(ActorCriticPolicy):
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


class PPOPolicy(PPONoScaleCNNFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                         cnn_extractor=extractor, **kwargs)


class SACNoScaleCNNFeedForwardPolicy(SACPolicy):
    """
        Adapted from stable_baselines.sac.policies.FeedForwardPolicy
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=None, reg_weight=0.0, layer_norm=False, act_fun=tf.nn.relu, **kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=False)

        self.layer_norm = layer_norm
        self.cnn_kwargs = kwargs
        if cnn_extractor is None:
            raise ValueError("CNN extractor cannot be None")
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        if layers is None:
            layers = [64, 64]
        self.layers = layers
        self.reg_loss = None
        self.reg_weight = reg_weight
        self.entropy = None

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ_fn = act_fun

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            self.act_mu = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)

        # Regularize policy output (not used for now)
        # reg_loss = self.reg_weight * 0.5 * tf.reduce_mean(log_std ** 2)
        # reg_loss += self.reg_weight * 0.5 * tf.reduce_mean(mu ** 2)
        # self.reg_loss = reg_loss

        # OpenAI Variation to cap the standard deviation
        # activation = tf.tanh # for log_std
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        # Original Implementation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        # MISSING: reg params for log and mu
        # Apply squashing and account for it in the probability
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn",
                     create_vf=True, create_qf=True):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)

            if create_vf:
                # Value function
                with tf.variable_scope('vf', reuse=reuse):
                    vf_h = mlp(critics_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    value_fn = tf.layers.dense(vf_h, 1, name="vf")
                self.value_fn = value_fn

            if create_qf:
                # Concatenate preprocessed state and action
                qf_h = tf.concat([critics_h, action], axis=-1)

                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf1 = tf.layers.dense(qf1_h, 1, name="qf1")

                with tf.variable_scope('qf2', reuse=reuse):
                    qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf2 = tf.layers.dense(qf2_h, 1, name="qf2")

                self.qf1 = qf1
                self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.act_mu, self.std], {self.obs_ph: obs})


class SACCustomPolicy(SACNoScaleCNNFeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, **kwargs):
        super().__init__(sess, ob_space, ac_space, cnn_extractor=extractor, **kwargs)
