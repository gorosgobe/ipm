import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import EarlyStopping
from ignite.metrics import Loss
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from lib.controller import TrainingPixelROI
from lib.dataset import ImageTipVelocitiesDataset
from lib.networks import *


class AlignmentLoss(_Loss):
    def __init__(self):
        super(AlignmentLoss, self).__init__()

    def forward(self, input, target):
        # noinspection PyTypeChecker
        similarity = torch.nn.functional.cosine_similarity(input, target, dim=1, eps=1e-8)
        clamped = torch.clamp(similarity, -1.0, 1.0)
        acos = torch.acos(clamped)
        return torch.mean(acos)


class TipVelocityEstimatorLoss(object):
    def __init__(self):
        self.mse_loss = torch.nn.MSELoss()
        self.alignment_loss = AlignmentLoss()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input, target):
        mse_loss = self.mse_loss(input, target)
        # alignment loss gives NANs for some reason
        # alignment_loss = self.alignment_loss(input, target)
        return mse_loss  # + alignment_loss


class TipVelocityEstimator(object):
    def __init__(self, batch_size, learning_rate, image_size, network_klass, transforms=None, name="model", device=None):
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.image_size = image_size
        width, height = self.image_size
        self.network = network_klass(width, height)

        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_func = TipVelocityEstimatorLoss()

        # set in train()
        self.train_data_loader = None
        self.val_loader = None

        self.training_losses = []
        self.validation_losses = []
        self.best_val_loss = None
        # stores best model
        self.best_info = None
        self.test_loss = None

        self.resize_transform = ResizeTransform(self.image_size)
        # transformations applied to input, except for initial resize
        self.transforms = transforms

        self.trainer = self._create_trainer()
        self.training_evaluator = self._create_evaluator()
        self.validation_evaluator = self._create_evaluator(early_stopping=True, patience=10)

    def train(self, data_loader, max_epochs, validate_epochs=10, val_loader=None):
        """
        validate_epochs: After how many epochs do we want to 
        validate our model against the validation dataset
        """
        self.train_data_loader = data_loader
        self.validate_epochs = validate_epochs
        self.val_loader = val_loader
        return self.trainer.run(data_loader, max_epochs)

    def get_network(self):
        return self.network

    @staticmethod
    def prepare_batch(batch, device, non_blocking):
        predictable = batch["tip_velocities"]
        if "rotations" in batch:
            # concatenate rotations to get 6-dimensional prediction
            predictable = torch.cat((predictable, batch["rotations"]), 1)
        return batch["image"].to(device), predictable.to(device)

    def epoch_started(self):
        def static_epoch_started(trainer):
            print("Epoch {}".format(trainer.state.epoch))

        return static_epoch_started

    def epoch_completed(self):
        def static_epoch_completed(trainer):
            self.training_evaluator.run(self.train_data_loader)
            metrics = self.training_evaluator.state.metrics
            loss = metrics["loss"]
            self.training_losses.append((trainer.state.epoch, loss))
            print("Training loss {}".format(loss))

        return static_epoch_completed

    def epoch_validate(self):
        def static_epoch_validate(trainer):
            if self.val_loader is not None and trainer.state.epoch % self.validate_epochs == 0:
                # validate against validation dataset
                self.validation_evaluator.run(self.val_loader)
                metrics = self.validation_evaluator.state.metrics
                loss = metrics["loss"]
                self.validation_losses.append((trainer.state.epoch, loss))
                print("Validation loss: {}".format(loss))
                # save according to best validation loss

                if self.best_val_loss is None or (self.best_val_loss is not None and loss < self.best_val_loss):
                    self.best_val_loss = loss
                    # store test loss information too
                    # like this, every saved model has information about its test loss
                    test_loss = self.evaluate_test(test_data_loader)
                    self.test_loss = test_loss
                    print("Test loss: ", test_loss)
                    # Best model is saved at end of training to minimise number of models saved (not really
                    # checkpointing)
                    self.best_info = self.get_info()

        return static_epoch_validate

    def _create_trainer(self):
        trainer = create_supervised_trainer(
            self.network,
            self.optimiser,
            self.loss_func,
            device=self.device,
            prepare_batch=TipVelocityEstimator.prepare_batch
        )
        trainer.add_event_handler(Events.EPOCH_STARTED, self.epoch_started())
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed())
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_validate())
        return trainer

    def _create_evaluator(self, early_stopping=False, patience=10):
        evaluator = create_supervised_evaluator(self.network, device=self.device, metrics={
            "loss": Loss(self.loss_func)
        }, prepare_batch=TipVelocityEstimator.prepare_batch)

        if early_stopping:
            def score_function(engine):
                val_loss = engine.state.metrics["loss"]
                return -val_loss

            early_stopping = EarlyStopping(
                patience=patience,
                score_function=score_function,
                trainer=self.trainer
            )
            evaluator.add_event_handler(Events.COMPLETED, early_stopping)

        return evaluator

    def get_info(self):
        return {
            "model_state_dict": self.network.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "epoch": self.trainer.state.epoch if self.trainer.state is not None else 0,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "transforms": self.transforms,
            "image_size": self.image_size,
            "test_loss": self.test_loss,
            "name": self.name,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "network_klass": type(self.network)
        }

    def save(self, path, info=None):
        if info is None:
            torch.save(self.get_info(), path)
        else:
            torch.save(info, path)

    def save_best_model(self, path):
        best_val_loss = self.best_info["validation_losses"][-1][1]
        best_epoch = self.best_info["epoch"]
        name = self.best_info["name"]
        file_name = "{}_{}_{}.pt".format(name, best_epoch, best_val_loss)
        self.save(os.path.join(path, file_name), self.best_info)

    def load_parameters(self, state_dict):
        self.network.load_state_dict(state_dict)

    def load_optimiser_parameters(self, state_dict):
        self.optimiser.load_state_dict(state_dict)

    @staticmethod
    def load(path):
        info = torch.load(path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        batch_size = info["batch_size"]
        learning_rate = info["learning_rate"]
        state_dict = info["model_state_dict"]
        optimiser_state_dict = info["optimiser_state_dict"]
        transforms = info["transforms"]
        image_size = info["image_size"]
        epoch = info["epoch"]
        name = info["name"]
        network_klass = info["network_klass"]

        estimator = TipVelocityEstimator(batch_size, learning_rate, image_size, network_klass, transforms, name)
        estimator.test_loss = info["test_loss"]

        def setup_state(engine):
            engine.state.epoch = epoch

        estimator.trainer.add_event_handler(Events.STARTED, setup_state)

        estimator.training_losses = info["training_losses"]
        estimator.validation_losses = info["validation_losses"]

        estimator.load_parameters(state_dict)
        estimator.load_optimiser_parameters(optimiser_state_dict)

        return estimator

    def resize_image(self, image):
        """
        Resize image to size of image that the controller was trained with
        """
        return self.resize_transform(image)

    def predict(self, batch):
        with torch.no_grad():
            return self.network.forward(batch)

    def evaluate_test(self, test_loader):
        test_evaluator = self._create_evaluator()
        test_evaluator.run(test_loader)
        return test_evaluator.state.metrics["loss"]

    def plot_train_val_losses(self):
        training_epochs, training_losses = zip(*self.training_losses)
        plt.plot(training_epochs, training_losses, label="Train loss")
        validation_epochs, validation_losses = zip(*self.validation_losses)
        plt.plot(validation_epochs, validation_losses, label="Validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.show()


class ResizeTransform(object):
    def __init__(self, size):
        self.size = size  # tuple, like (128, 96)

    def __call__(self, image):
        return cv2.resize(image, dsize=self.size)


if __name__ == "__main__":
    config = dict(
        seed=2019,
        # if pixel cropper is used to decrease size by two in both directions, size has to be decreased accordingly
        # otherwise we would be feeding a higher resolution cropped image
        # we want to supply a cropped image, corresponding exactly to the resolution of that area in the full image
        size=(128, 96),
        velocities_csv="text_camera_orient/velocities.csv",
        rotations_csv="text_camera_orient/rotations.csv",
        metadata="text_camera_orient/metadata.json",
        root_dir="text_camera_orient",
        initial_pixel_cropper=None, #TrainingPixelROI(480 // 2, 640 // 2),  # set to None for full image initially
        cache_images=False,
        batch_size=32,
        split=[0.8, 0.1, 0.1],
        name="TESTING_ORIENTATIONS",
        learning_rate=0.0001,
        max_epochs=5,
        validate_epochs=1,
        save_to_location="models/",
        network_klass=FullImageNetwork,
    )

    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # set up GPU if available
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.manual_seed(config["seed"])
    print("Using GPU: {}".format(use_cuda))

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    preprocessing_transforms = torchvision.transforms.Compose([ResizeTransform(config["size"]),
                                                               transforms
                                                               ])

    dataset = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        initial_pixel_cropper=config["initial_pixel_cropper"],
        transform=preprocessing_transforms,
        cache_images=config["cache_images"],
    )

    total_demonstrations = dataset.get_num_demonstrations()

    training_demonstrations, n_training_dems = dataset.get_split(config["split"][0], total_demonstrations, 0)
    val_demonstrations, n_val_dems = dataset.get_split(config["split"][1], total_demonstrations, n_training_dems)
    test_demonstrations, n_test_dems = dataset.get_split(config["split"][2], total_demonstrations, n_training_dems + n_val_dems)

    # Limited dataset
    #training_demonstrations, n_training_dems = dataset.get_split(0.2, total_demonstrations, 0)

    print("Training demonstrations: ", n_training_dems, len(training_demonstrations))
    print("Validation demonstrations: ", n_val_dems, len(val_demonstrations))
    print("Test demonstrations: ", n_test_dems, len(test_demonstrations))

    train_data_loader = DataLoader(training_demonstrations, batch_size=config["batch_size"], num_workers=8,
                                   shuffle=True)
    validation_data_loader = DataLoader(val_demonstrations, batch_size=4, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_demonstrations, batch_size=4, num_workers=8, shuffle=True)

    tip_velocity_estimator = TipVelocityEstimator(
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        image_size=config["size"],
        network_klass=config["network_klass"],
        # transforms without initial resize, so they can be pickled correctly
        transforms=transforms,
        name=config["name"],
        device=device
    )

    tip_velocity_estimator.train(
        train_data_loader,
        max_epochs=config["max_epochs"],  # or stop early with patience 10
        validate_epochs=config["validate_epochs"],
        val_loader=validation_data_loader
    )

    # save_best_model
    tip_velocity_estimator.save_best_model(config["save_to_location"])
    #tip_velocity_estimator.plot_train_val_losses()
