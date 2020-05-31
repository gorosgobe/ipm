import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from lib.common.saveable import Saveable
from lib.common.utils import get_demonstrations, ResizeTransform
from lib.cv.controller import TrainingPixelROI
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.dsae.dsae_dataset import DSAE_FeatureCropTVEAdapter, DSAE_SingleFeatureProviderDataset
from lib.networks import AttentionNetworkCoordGeneral


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

    def get_best_feature_index(self, crop_size=(32, 24)):
        validation_losses = []
        best_val_loss = None
        best_estimator = None
        for idx, f in enumerate(range(self.features)):
            val_loss, estimator = self.train_model_with_feature(index=idx, crop_size=crop_size)

            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss, best_estimator = (val_loss, estimator)

            print(f"Index {idx}, val loss {val_loss}")
            validation_losses.append(val_loss)

        self.losses = validation_losses
        res = np.array(validation_losses).argmin()
        self.index = res
        self.best_estimator = best_estimator
        return res

    def get_index(self):
        return self.index

    @staticmethod
    def load_info(path):
        return torch.load(path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def save_estimator(self, path):
        self.best_estimator.save_best_model(path=path)

    def save(self, path, info=None):
        # save best estimator as well as extra information
        self.save_estimator(path=path)
        super().save(path=os.path.join(path, self.name), info=info)

    def get_info(self):
        return dict(
            name=self.name,
            validation_losses=self.losses,
            index=self.index
        )


class DSAE_ChooserROI(object):
    def __init__(self, chooser_index, feature_provider, chooser_crop_size=(32, 24), size=(128, 96)):
        self.chooser_index = chooser_index
        self.chooser_crop_size = chooser_crop_size
        self.feature_provider = feature_provider  # returns (B, C * 2)
        self.size = size
        self.resize_transform = ResizeTransform(size=size)
        self.feature_provider_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        cropped_width, cropped_height = chooser_crop_size
        self.pixel_cropper = TrainingPixelROI(
            cropped_height=cropped_height, cropped_width=cropped_width, add_spatial_maps=True
        )
        self.cropped_np_image_normalisation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5])
        ])

    def crop(self, image):
        # takes (640, 480) np image, should return (32, 24) cropped np image
        image = self.resize_transform(image)
        # convert to torch, perform normalisation and feed through autoencoder to get features
        torch_img = self.feature_provider_transform(image)
        features = self.feature_provider(torch_img.unsqueeze(0)).squeeze(0).view(-1, 2)
        # get pixel from features
        selected_feature = features[self.chooser_index]
        selected_feature = (selected_feature + 1) / 2
        w, h = self.size
        pixel = (selected_feature * torch.tensor([w - 1, h - 1], dtype=torch.float32)).type(dtype=torch.int32)
        # crop to (32, 24) based on pixel, convert image to torch and normalise
        # normalisation for DSAE Chooser is applied at the dataset level
        cropped_image, _ = self.pixel_cropper.crop(image, pixel)
        return self.cropped_np_image_normalisation(cropped_image), None
