import os

import matplotlib.pyplot as plt
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import EarlyStopping
from ignite.metrics import Loss
from torch.nn.modules.loss import _Loss

from lib.networks import *
from lib.utils import ResizeTransform


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

    def train(self, data_loader, max_epochs, val_loader, test_loader, validate_epochs=10):
        """
        validate_epochs: After how many epochs do we want to 
        validate our model against the validation dataset
        """
        self.train_data_loader = data_loader
        self.validate_epochs = validate_epochs
        self.val_loader = val_loader
        self.test_loader = test_loader
        return self.trainer.run(data_loader, max_epochs)

    def get_network(self):
        return self.network

    @staticmethod
    def prepare_batch(batch, device, non_blocking):
        input_data = batch["image"]
        predictable = batch["tip_velocities"]
        if "rotations" in batch:
            # concatenate rotations to get 6-dimensional prediction
            predictable = torch.cat((predictable, batch["rotations"]), 1)

        # case where we want to estimate from relative quantities, rather than image
        if "relative_target_position" in batch and "relative_target_orientation" in batch:
            input_data = torch.cat((batch["relative_target_position"], batch["relative_target_orientation"]), dim=1)
        elif "pixel_info" in batch:
            # case where pixels are used, cropped version
            input_data = (
                batch["image"].to(device),
                batch["pixel_info"]["top_left"].to(device),
                batch["pixel_info"]["bottom_right"].to(device),
                batch["pixel_info"]["original_image_width"].to(device),
                batch["pixel_info"]["original_image_height"].to(device),
            )

        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(device)

        if isinstance(predictable, torch.Tensor):
            predictable = predictable.to(device)

        # otherwise, we assume the individual components have already sent to the GPU, if we set input to a pair,
        # for example, for convenience
        return input_data, predictable

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
            if trainer.state.epoch % self.validate_epochs == 0:
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
                    test_loss = self.evaluate_test(self.test_loader)
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