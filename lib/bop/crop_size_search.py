from ax import optimize
from ax.storage.json_store.encoder import object_to_json

from lib.common.saveable import Saveable


class CropSizeSearch(Saveable):
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
        self.model_json = None

    def evaluation_function(self, parameterization):
        min_width, min_height = self.min
        max_width, max_height = self.max
        width = round(min_width + (max_width - min_width) * parameterization["x"])
        height = round(min_height + (max_height - min_height) * parameterization["y"])
        val_loss = self.dsae_feature_chooser.train_model_with_feature(
            index=parameterization["feature"],
            crop_size=(width, height)
        )

        return dict(val_loss=val_loss)

    def search(self, total_trials):
        best_parameters, values, experiment, model = optimize(
            parameters=[
                {"name": "feature", "type": "choice", "values": list(range(self.features))},
                {"name": "width", "type": "range", "bounds": [0.0, 1.0]},
                {"name": "height", "type": "range", "bounds": [0.0, 1.0]}
            ],
            evaluation_function=lambda params: self.evaluation_function(params),
            objective_name="val_loss",
            total_trials=total_trials
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
        self.model_json = object_to_json(model)

    def get_info(self):
        return dict(
            name=self.name,
            best_parameters=self.best_parameters,
            values=self.values,
            experiment=self.experiment,
            model_json=self.model_json
        )
