import numpy as np


class MetaTrainingEvaluator(object):
    def __init__(self, test_env, num_iter):
        self.test_env = test_env
        self.num_iter = num_iter
        self.rl_model = None

    def set_rl_model(self, rl_model):
        self.rl_model = rl_model

    def evaluate_and_get(self):
        raise NotImplementedError("evaluate_and_get needs to be implemented")

    @staticmethod
    def reset_and_run(rl_model, test_env, num_iter):
        if rl_model is None:
            raise ValueError("You forgot to set the RL model!")

        obs = test_env.reset(num_demonstrations=num_iter)
        done = False
        while not done:
            action, _states = rl_model.predict(obs, deterministic=True)
            obs, _, done, info = test_env.step(action)

    @staticmethod
    def random_and_run(test_env, num_iter):
        _obs = test_env.reset(num_demonstrations=num_iter)
        done = False
        while not done:
            action = np.random.uniform(-1.0, 1.0, size=2)
            _obs, _, done, info = test_env.step(action)


class CropEvaluator(MetaTrainingEvaluator):
    def __init__(self, test_env, num_iter=1):
        super().__init__(test_env, num_iter)

    def evaluate_and_get(self):
        MetaTrainingEvaluator.reset_and_run(self.rl_model, self.test_env, self.num_iter)
        return self.test_env.get_processed_data_from_states()


class RandomCropEvaluator(MetaTrainingEvaluator):
    def __init__(self, test_env, num_iter=1):
        super().__init__(test_env, num_iter)

    def evaluate_and_get(self):
        MetaTrainingEvaluator.random_and_run(self.test_env, self.num_iter)
        return self.test_env.get_processed_data_from_states()


class FilterSpatialEvaluator(MetaTrainingEvaluator):
    def __init__(self, test_env, num_iter=1):
        super().__init__(test_env, num_iter)

    # returns top k selected features for every timestep
    def evaluate_and_get(self):
        MetaTrainingEvaluator.reset_and_run(self.rl_model, self.test_env, self.num_iter)
        return self.test_env.get_selected_features(), self.test_env.get_target_predictions()
