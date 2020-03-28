import enum

import torch
import torchmeta
from lib.meta.pytorch_maml.maml.metalearners import MAML
from metalearners import FOMAML, MetaSGD


class MetaAlgorithm(enum.Enum):
    MAML = 0
    FOMAML = 1
    METASGD = 2


class MetaImitationLearning(object):
    def __init__(self, model, meta_algorithm, num_adaptation_steps, step_size, optimizer, loss_function,
                 max_batches=100, scheduler=None, device=None):
        # num_adaptation_steps: number of steps for inner loop
        self.model = model
        self.max_batches = max_batches  # Number of batches of tasks per epoch
        params = dict(
            model=model,
            optimizer=optimizer,
            num_adaptation_steps=num_adaptation_steps,
            scheduler=scheduler,
            loss_function=loss_function,
            step_size=step_size,
            device=device
        )
        if meta_algorithm == MetaAlgorithm.FOMAML:
            self.maml = FOMAML(**params)
        elif meta_algorithm == MetaAlgorithm.MAML:
            self.maml = MAML(**params)
        elif meta_algorithm == MetaAlgorithm.METASGD:
            # in MetaSGD, step size is initial step size to start learning from
            params["init_step_size"] = params["step_size"]
            del params["step_size"]
            self.maml = MetaSGD(**params)

    def train(self, train_batch_dataloader, val_batch_dataloader, num_epochs):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            self.maml.train(train_batch_dataloader, max_batches=self.max_batches, verbose=True, desc="Training",
                            leave=False)
            results = self.maml.evaluate(val_batch_dataloader, max_batches=self.max_batches, verbose=True,
                                         desc=f"Validation for epoch {epoch + 1}")
            print(f"Mean outer loss: {results['mean_outer_loss']}")

    def get_model(self):
        return self.model

    def test(self):
        pass

