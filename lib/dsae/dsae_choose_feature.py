import numpy as np
from torch.utils.data import DataLoader

from dsae.dsae_dataset import DSAE_FeatureCropTVEAdapter, DSAE_SingleFeatureProviderDataset
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.networks import AttentionNetworkCoordGeneral


class DSAE_ValFeatureChooser(object):
    # Uses heuristic: pick feature such that when we crop around it, it gives us the
    # best performance on a validation set
    def __init__(self, latent_dimension, feature_provider_dataset, device):
        self.features = latent_dimension // 2
        self.device = device
        # TODO: split here
        self.training_dataset = feature_provider_dataset
        self.validation_dataset = feature_provider_dataset

    def train_model_with_feature(self, index, crop_size=(32, 24)):
        estimator = TipVelocityEstimator(
            batch_size=32,
            learning_rate=0.0001,
            image_size=crop_size,
            network_klass=AttentionNetworkCoordGeneral.create(*crop_size),
            device=self.device
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

        train_dataloader = DataLoader(dataset=training_dataset, batch_size=32, num_workers=2)
        val_dataloader = DataLoader(dataset=validation_dataset, batch_size=32, num_workers=2)

        estimator.train(data_loader=train_dataloader, max_epochs=200, val_loader=val_dataloader, validate_epochs=1)
        val_loss = estimator.get_best_val_loss()
        # crop resolution
        return val_loss

    def get_best_feature_index(self):
        validation_losses = []
        for idx, f in enumerate(len(self.features)):
            val_loss = self.train_model_with_feature(index=idx)
            validation_losses.append(val_loss)

        return np.array(validation_losses).argmin()
