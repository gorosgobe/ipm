import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.common.utils import get_demonstrations
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.dsae.dsae_dataset import DSAE_FeatureCropTVEAdapter, DSAE_SingleFeatureProviderDataset
from lib.networks import AttentionNetworkCoordGeneral
from lib.common.saveable import Saveable


class DSAE_ValFeatureChooser(Saveable):
    # Uses heuristic: pick feature from DSAE such that when we crop around it, it gives us the best performance on a
    # validation set
    def __init__(self, name, latent_dimension, feature_provider_dataset, split, device, limit_train_coeff=-1):
        super().__init__()
        super().__init__()
        self.name = name
        self.features = latent_dimension // 2
        self.device = device
        self.training_dataset, self.validation_dataset, _ = get_demonstrations(
            dataset=feature_provider_dataset,
            split=split,
            limit_train_coeff=limit_train_coeff
        )
        self.losses = []
        self.index = None
        self.best_estimator = None

    def train_model_with_feature(self, index, crop_size=(32, 24)):
        estimator = TipVelocityEstimator(
            name=f"{self.name}_model",
            batch_size=32,
            learning_rate=0.0001,
            image_size=crop_size,
            network_klass=AttentionNetworkCoordGeneral.create(*crop_size),
            device=self.device,
            verbose=False
        )

        training_dataset = DSAE_FeatureCropTVEAdapter(
            single_feature_dataset=DSAE_SingleFeatureProviderDataset(
                feature_provider_dataset=self.training_dataset,
                feature_index=index
            ),
            crop_size=crop_size
        )

        validation_dataset = DSAE_FeatureCropTVEAdapter(
            single_feature_dataset=DSAE_SingleFeatureProviderDataset(
                feature_provider_dataset=self.validation_dataset,
                feature_index=index
            ),
            crop_size=crop_size
        )

        train_dataloader = DataLoader(dataset=training_dataset, batch_size=32, num_workers=4, shuffle=True)
        val_dataloader = DataLoader(dataset=validation_dataset, batch_size=32, num_workers=4, shuffle=True)

        estimator.train(data_loader=train_dataloader, max_epochs=200, val_loader=val_dataloader, validate_epochs=1)
        val_loss = estimator.get_best_val_loss()
        return val_loss, estimator

    def get_best_feature_index(self):
        validation_losses = []
        best_val_loss = None
        best_estimator = None
        for idx, f in enumerate(range(self.features)):
            val_loss, estimator = self.train_model_with_feature(index=idx)
            best_val_loss, best_estimator = val_loss, estimator if best_val_loss is None or val_loss < best_val_loss else \
                (best_val_loss, best_estimator)
            print(f"Index {idx}, val loss {val_loss}")
            validation_losses.append(val_loss)

        self.losses = validation_losses
        res = np.array(validation_losses).argmin()
        self.index = res
        self.best_estimator = best_estimator
        return res

    @staticmethod
    def load_info(path):
        return torch.load(path)

    def save_estimator(self, path):
        self.best_estimator.save_best_model(path=path)

    def save(self, path, info=None):
        # save best estimator as well as extra information
        self.save_estimator(path=path)
        super().save(path=path, info=info)

    def get_info(self):
        return dict(
            name=self.name,
            validation_losses=self.losses,
            index=self.index
        )
