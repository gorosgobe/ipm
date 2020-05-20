from collections import OrderedDict

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from lib.common.saveable import Saveable


class STNManager(Saveable):
    def __init__(self, name, stn, device, loc_lr=1e-3, model_lr=1e-2):
        super().__init__()
        self.name = name
        self.stn = stn
        self.best_info = None
        self.device = device
        self.loss = nn.MSELoss()
        self.stn_optimiser = torch.optim.SGD(self.stn.localisation_param_regressor.parameters(), lr=loc_lr)
        self.model_optimiser = torch.optim.SGD(self.stn.model.parameters(), lr=model_lr)
        self.loc_lr = loc_lr
        self.model_lr = model_lr
        self.stn_scheduler = StepLR(self.stn_optimiser, step_size=20, gamma=0.1)
        self.model_scheduler = StepLR(self.model_optimiser, step_size=20, gamma=0.1)

    def get_loss(self, batch, params=None):
        images = batch["image"].to(self.device)
        tip_velocities = batch["tip_velocities"].to(self.device)
        rotations = batch["rotations"].to(self.device)
        targets = torch.cat((tip_velocities, rotations), 1)
        predicted_targets = self.stn(images, params=params)
        return self.loss(predicted_targets, targets)

    @staticmethod
    def pseudo_infinite_sampling(dataloader, max_count):
        count = 0
        while True and count < max_count:
            for idx, batch in enumerate(dataloader):
                yield batch
                count += 1

    def train(self, num_epochs, train_dataloader, val_dataloader, test_dataloader):
        self.stn.to(self.device)
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            self.stn.localisation_param_regressor.train()
            self.stn.model.train()
            # one epoch of training for regression model with default transformation (set to identity at the beginning)
            train_loss_epoch = 0
            for batch_idx, batch in enumerate(train_dataloader):
                self.stn_optimiser.zero_grad()
                self.model_optimiser.zero_grad()
                loss = self.get_loss(batch)
                train_loss_epoch += loss.item()
                loss.backward()
                self.model_optimiser.step()
            print("Train loss", train_loss_epoch / len(train_dataloader.dataset))

            # find parameters theta prime from training data
            for train_batch, val_batch in zip(
                    self.pseudo_infinite_sampling(train_dataloader, len(train_dataloader)),
                    self.pseudo_infinite_sampling(val_dataloader, len(train_dataloader))
            ):
                # we now have training and validation batches
                # usually fewer validation batches than training, so reiterate over val loader
                self.stn_optimiser.zero_grad()
                self.model_optimiser.zero_grad()
                train_loss = self.get_loss(train_batch)

                # theta
                fast_weights = OrderedDict((name, param) for (name, param) in self.stn.model.named_parameters())
                # gradient with respect to regression network, retain it in the graph
                grads = torch.autograd.grad(train_loss, self.stn.model.parameters(), create_graph=True)
                # theta prime
                fast_weights = OrderedDict(
                    (name, param - self.model_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))

                # compute val loss using theta prime
                val_loss = self.get_loss(val_batch, params=fast_weights)
                val_loss.backward()
                self.stn_optimiser.step()

            self.stn_scheduler.step()
            self.model_scheduler.step()
            # evaluate on val and test
            self.stn.model.eval()
            self.stn.localisation_param_regressor.eval()
            with torch.no_grad():
                val_eval_loss_epoch = 0
                for _, val_eval_batch in enumerate(val_dataloader):
                    val_eval_loss = self.get_loss(val_eval_batch)
                    val_eval_loss_epoch += val_eval_loss.item()
                print("Val EVAL:", val_eval_loss_epoch / len(val_dataloader.dataset))

                test_eval_loss_epoch = 0
                for _, test_eval_batch in enumerate(test_dataloader):
                    test_eval_loss = self.get_loss(test_eval_batch)
                    test_eval_loss_epoch += test_eval_loss.item()
                print("Test EVAL:", test_eval_loss_epoch / len(test_dataloader.dataset))
                # TODO: early stopping on test loss, then evaluate on trajectories

    def get_info(self):
        return dict(
            name=self.name,
            stn_state_dict=self.stn.state_dict()
        )
