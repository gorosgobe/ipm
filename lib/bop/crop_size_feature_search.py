import numpy as np
from ax import optimize
from ax.plot.contour import plot_contour
from ax.plot.render import plot_config_to_html
from ax.plot.trace import optimization_trace_single_method
from ax.storage.json_store.encoder import object_to_json

from lib.common.saveable import Saveable


class CropSizeFeatureSearch(Saveable):
    def __init__(self, name, latent_dimension, dsae_feature_chooser, max=(128, 96), min=(32, 24)):
        super().__init__()
        self.name = name
        self.features = latent_dimension // 2
        self.dsae_feature_chooser = dsae_feature_chooser
        self.max = max
        self.min = min

        self.best_parameters = None
        self.values = None
        self.experiment = None
        self.best_plot_contour = None
        self.improvement_plot = None

    def evaluation_function(self, parameterization):
        min_width, min_height = self.min
        max_width, max_height = self.max
        width = round(min_width + (max_width - min_width) * parameterization["width"])
        height = round(min_height + (max_height - min_height) * parameterization["height"])
        index = parameterization["feature"]
        print(f"Trial for feature {index}, size ({width}, {height})")
        val_loss = self.dsae_feature_chooser.train_model_with_feature(
            index=parameterization["feature"],
            crop_size=(width, height)
        )

        return val_loss

    def search(self, total_trials):
        best_parameters, values, experiment, model = optimize(
            parameters=[
                {"name": "feature", "type": "choice", "values": list(range(self.features))},
                {"name": "width", "type": "range", "bounds": [0.0, 1.0]},
                {"name": "height", "type": "range", "bounds": [0.0, 1.0]}
            ],
            evaluation_function=lambda params: self.evaluation_function(params),
            total_trials=total_trials,
            minimize=True
        )

        print("Best parameters:")
        print(best_parameters)
        print("Means:")
        means, covariances = values
        print(means)
        print("Covariances:")
        print(covariances)

        self.best_parameters = best_parameters
        self.values = values
        self.experiment = object_to_json(experiment)
        # cannot store model directly
        best_contour_config = plot_contour(model=model, param_x="width", param_y="height", metric_name='hartmann6',
                                           slice_values=dict(
                                               feature=best_parameters["feature"]
                                           ))
        self.best_plot_contour = plot_config_to_html(best_contour_config)
        best_objectives = np.array(
            [[trial.objective_mean * 100 for trial in experiment.trials.values()]])
        improvement_plot = optimization_trace_single_method(
            y=np.maximum.accumulate(best_objectives, axis=1),
            title="Validation loss vs. # of iterations",
            ylabel="Validation loss",
        )
        self.improvement_plot = plot_config_to_html(improvement_plot)

    def get_info(self):
        return dict(
            name=self.name,
            best_parameters=self.best_parameters,
            values=self.values,
            experiment=self.experiment,
            best_plot_contour=self.best_plot_contour,
            improvement_plot=self.improvement_plot
        )

    def save_plots(self, path):
        with open(path + "_contour.html", "w+") as f:
            f.write(self.best_plot_contour)

        with open(path + "_improvement.html", "w+") as f:
            f.write(self.improvement_plot)
