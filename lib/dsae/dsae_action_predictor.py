import torch
from torch import nn

from lib.dsae.dsae_networks import DSAE_TargetActionPredictor
from lib.common.early_stopper import EarlyStopper
from lib.common.saveable import BestSaveable


class ActionPredictorManager(BestSaveable):
    def __init__(self, action_predictor, num_epochs, optimiser, device, name="model", verbose=False):
        super().__init__()
        self.action_predictor = action_predictor
        self.num_epochs = num_epochs
        self.name = name
        self.device = device
        self.optimiser = optimiser
        self.loss = nn.MSELoss()
        self.best_info = None
        self.verbose = verbose
        self.early_stopper = EarlyStopper(patience=10, saveable=self)

    def get_loss(self, batch):
        # we do not have images here, as features are provided directly from the RL agent
        # they are obtained previously from another dataset and a feature provider (i.e. some sort of DSAE)
        features = batch["features"].to(self.device)
        targets = batch["target_vel_rot"].to(self.device)
        predictions = self.action_predictor(features)
        return self.loss(input=predictions, target=targets)

    def train(self, train_dataloader, validation_dataloader):
        self.action_predictor.to(self.device)
        for epoch in range(self.num_epochs):
            train_loss_epoch = 0
            self.action_predictor.train()
            for batch_idx, batch in enumerate(train_dataloader):
                self.optimiser.zero_grad()
                loss = self.get_loss(batch)
                train_loss_epoch += loss.item()
                loss.backward()
                self.optimiser.step()

            train_loss_epoch = train_loss_epoch / len(train_dataloader.dataset)
            if self.verbose:
                print(f"Epoch {epoch + 1}: {train_loss_epoch}")

            self.action_predictor.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                for val_batch_idx, val_batch in enumerate(validation_dataloader):
                    val_loss = self.get_loss(val_batch)
                    val_loss_epoch += val_loss.item()

                val_loss_epoch = val_loss_epoch / len(validation_dataloader.dataset)
                if self.verbose:
                    print("Validation loss", val_loss_epoch)
                self.early_stopper.register_loss(val_loss_epoch)

            if self.early_stopper.should_stop():
                if self.verbose:
                    print("Patience reached, stopping...")
                break

    def get_validation_loss(self):
        return self.early_stopper.get_best_val_loss()

    @staticmethod
    def load(path, k):
        info = torch.load(path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model = DSAE_TargetActionPredictor(k=k)
        model.load_state_dict(info["state_dict"])
        return model

    def get_info(self):
        return dict(
            state_dict=self.action_predictor.state_dict(),
            name=self.name
        )

    def get_best_info(self):
        return self.best_info
