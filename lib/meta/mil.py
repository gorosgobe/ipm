import enum

import numpy as np
import torch

from maml.metalearners.maml import FOMAML, MAML
from maml.metalearners.meta_sgd import MetaSGD
from lib.common.saveable import BestSaveable


class MetaAlgorithm(enum.Enum):
    MAML = 0
    FOMAML = 1
    METASGD = 2


class MetaImitationLearning(BestSaveable):
    def __init__(self, name, model, meta_algorithm, num_adaptation_steps, step_size, optimiser, loss_function,
                 max_batches=100, scheduler=None, device=None):
        super().__init__()
        # num_adaptation_steps: number of steps for inner loop
        self.name = name
        self.model = model
        self.max_batches = max_batches  # Number of batches of tasks per epoch
        self.mean_outer_train_losses = []
        self.mean_outer_val_losses = []
        self.best_info = None
        self.params = dict(
            model=model,
            optimizer=optimiser,
            num_adaptation_steps=num_adaptation_steps,
            scheduler=scheduler,
            loss_function=loss_function,
            step_size=step_size,
            device=device
        )
        if meta_algorithm == MetaAlgorithm.FOMAML:
            self.maml = FOMAML(**self.params)
        elif meta_algorithm == MetaAlgorithm.MAML:
            self.maml = MAML(**self.params)
        elif meta_algorithm == MetaAlgorithm.METASGD:
            # in MetaSGD, step size is initial step size to start learning from
            copy_params = dict(**self.params)
            copy_params["init_step_size"] = copy_params["step_size"]
            del copy_params["step_size"]
            self.maml = MetaSGD(**copy_params)

    def train(self, train_batch_dataloader, val_batch_dataloader, num_epochs):
        self.best_info = self.get_info()
        best_val_loss = None
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            train_outer_losses = []
            for train_results in self.maml.train_iter(train_batch_dataloader, max_batches=self.max_batches):
                train_outer_losses.append(train_results["mean_outer_loss"])

            mean_outer_train_loss = np.mean(np.array(train_outer_losses))
            print("Training outer loss:", mean_outer_train_loss)
            self.mean_outer_train_losses.append(mean_outer_train_loss)

            results = self.maml.evaluate(val_batch_dataloader, max_batches=self.max_batches)
            val_mean_outer_loss = results['mean_outer_loss']
            print(f"Validation outer loss: {val_mean_outer_loss}")
            self.mean_outer_val_losses.append(val_mean_outer_loss)

            # store best information when val loss is the best
            if best_val_loss is None:
                best_val_loss = val_mean_outer_loss
                self.best_info = self.get_info()
            elif best_val_loss is not None and val_mean_outer_loss < best_val_loss:
                best_val_loss = val_mean_outer_loss
                self.best_info = self.get_info()

    def get_model(self):
        return self.model

    @staticmethod
    def load_best_params(path):
        info = torch.load(path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # if best saved, these are the best parameters according to the validation tasks
        # ATM we only support loading the pretrained parameters :)
        return info["model_state_dict"]

    def get_info(self):
        return dict(
            model_state_dict=self.model.state_dict(),
            optimiser_state_dict=self.params["optimizer"].state_dict(),
            num_adaptation_steps=self.params["num_adaptation_steps"],
            network_klass=type(self.model),
            name=self.name,
            step_size=self.params["step_size"],
            mean_outer_train_losses=self.mean_outer_train_losses,
            mean_outer_val_losses=self.mean_outer_val_losses
        )

    def get_best_info(self):
        return self.best_info
