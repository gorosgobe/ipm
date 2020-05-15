import random

import matplotlib.pyplot as plt
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import EarlyStopping
from ignite.metrics import Loss

from lib.networks import *
from lib.common.utils import ResizeTransform
from lib.common.saveable import BestSaveable


class AlignmentLoss(object):
    def __call__(self, input, target):
        # noinspection PyTypeChecker
        eps = 1e-7
        similarity = torch.nn.functional.cosine_similarity(input, target, dim=1, eps=eps)
        # torch.acos is not numerically very stable (https://github.com/pytorch/pytorch/issues/8069)
        # so clamp with epsilon
        clamped = torch.clamp(similarity, -1.0 + eps, 1.0 - eps)
        acos = torch.acos(clamped)
        return torch.mean(acos)


class TipVelocityEstimatorLoss(object):
    def __init__(self, is_composite_loss=False, mse_lambda=0.1, l1_lambda=1.0, alignment_lambda=0.005, verbose=True):
        self.mse_loss = torch.nn.MSELoss()

        # everything required when using composite loss from:
        # https://arxiv.org/abs/1710.04615
        # Deep Imitation Learning for Complex Manipulation Tasks from Virtual Reality Teleoperation
        self.is_composite_loss = is_composite_loss
        self.mse_lambda = mse_lambda
        self.l1_loss = torch.nn.L1Loss()
        self.l1_lambda = l1_lambda
        # alignment component
        self.alignment_loss = AlignmentLoss()
        self.alignment_lambda = alignment_lambda
        self.verbose = verbose

        if self.verbose:
            if self.is_composite_loss:
                print(
                    f"Composite L2 ({self.mse_lambda}), L1 ({self.l1_lambda}) and alignment loss ({self.alignment_lambda})")
            else:
                print("Standard L2 Loss")

    def __call__(self, input, target):
        mse_loss = self.mse_loss(input, target)
        if self.is_composite_loss:
            mse_loss *= self.mse_lambda
            mse_loss += self.l1_lambda * self.l1_loss(input, target)
            mse_loss += self.alignment_lambda * self.alignment_loss(input, target)
        return mse_loss


class TipVelocityEstimator(BestSaveable):
    def __init__(self, batch_size, learning_rate, image_size, network_klass=None, network=None, transforms=None, name="model", device=None,
                 patience=10, composite_loss_params=None, verbose=True, optimiser_params=None):
        super().__init__()
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.image_size = image_size

        # network
        width, height = self.image_size
        if network_klass is None and network is None:
            raise ValueError("Either give a network klass to be initialised, or provide an already initialised network")
        if network_klass is not None:
            self.network = network_klass(width, height)
        else:
            self.network = network

        # optimiser
        if optimiser_params is None:
            optimiser_params = dict(optim=torch.optim.Adam)
        optim = optimiser_params.pop("optim")
        self.optimiser = optim(self.network.parameters(), lr=learning_rate, **optimiser_params)

        self.composite_loss_params = composite_loss_params
        # For a composite loss, pass in the parameters as a dictionary
        if self.composite_loss_params is None:
            self.loss_func = TipVelocityEstimatorLoss(verbose=verbose)
        else:
            self.loss_func = TipVelocityEstimatorLoss(is_composite_loss=True, verbose=verbose, **self.composite_loss_params)

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
        self.validation_evaluator = self._create_evaluator(early_stopping=True, patience=self.patience)

        self.verbose = verbose

    def train(self, data_loader, max_epochs, val_loader, test_loader=None, validate_epochs=1):
        """
        validate_epochs: After how many epochs do we want to 
        validate our model against the validation dataset
        """
        self.train_data_loader = data_loader
        self.validate_epochs = validate_epochs
        self.val_loader = val_loader
        self.test_loader = test_loader
        return self.trainer.run(data_loader, max_epochs, seed=random.getrandbits(32))

    def get_network(self):
        return self.network

    @staticmethod
    def prepare_batch(batch, device, non_blocking):
        input_data = batch["image"] if "image" in batch else None

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
            if self.verbose:
                print("Epoch {}".format(trainer.state.epoch))

        return static_epoch_started

    def epoch_completed(self):
        def static_epoch_completed(trainer):
            self.training_evaluator.run(self.train_data_loader, seed=random.getrandbits(32))
            metrics = self.training_evaluator.state.metrics
            loss = metrics["loss"]
            self.training_losses.append((trainer.state.epoch, loss))
            if self.verbose:
                print("Training loss {}".format(loss))

        return static_epoch_completed

    def epoch_validate(self):
        def static_epoch_validate(trainer):
            if trainer.state.epoch % self.validate_epochs == 0:
                # validate against validation dataset
                self.validation_evaluator.run(self.val_loader, seed=random.getrandbits(32))
                metrics = self.validation_evaluator.state.metrics
                loss = metrics["loss"]
                self.validation_losses.append((trainer.state.epoch, loss))
                if self.verbose:
                    print("Validation loss: {}".format(loss))
                # save according to best validation loss

                if self.best_val_loss is None or (self.best_val_loss is not None and loss < self.best_val_loss):
                    self.best_val_loss = loss
                    # store test loss information too
                    if self.test_loader is not None:
                        # like this, every saved model has information about its test loss
                        test_loss = self.evaluate_test(self.test_loader)
                        self.test_loss = test_loss
                        if self.verbose:
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

    def get_best_info(self):
        return self.best_info

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

        estimator = TipVelocityEstimator(
            batch_size=batch_size, learning_rate=learning_rate, image_size=image_size, network_klass=network_klass,
            transforms=transforms, name=name
        )
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
        self.network.eval()
        with torch.no_grad():
            return self.network.forward(batch)

    def evaluate_test(self, test_loader):
        test_evaluator = self._create_evaluator()
        test_evaluator.run(test_loader, seed=random.getrandbits(32))
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

    def get_best_val_loss(self):
        _, validation_losses = zip(*self.validation_losses)
        return min(validation_losses)

    def get_val_losses(self):
        return self.validation_losses

    def get_num_epochs_trained(self):
        # does not take into account patience
        return len(self.training_losses)

    def start(self):
        # only required for recurrent controllers
        return
