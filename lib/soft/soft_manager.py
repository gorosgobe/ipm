import torch
from torch import nn

from lib.common.early_stopper import EarlyStopper
from lib.common.saveable import BestSaveable
from lib.soft.soft import SoftCNNLSTMNetwork


class SoftManager(BestSaveable):

    def __init__(self, name, dataset, device, hidden_size):
        super().__init__()
        self.name = name
        self.model = SoftCNNLSTMNetwork(hidden_size=hidden_size)
        self.dataset = dataset
        self.device = device
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss = nn.MSELoss(reduction="none")
        self.early_stopper = EarlyStopper(patience=10, saveable=self)
        self.best_info = None
        self.hidden_size = hidden_size

    def train(self, num_epochs, train_dataloader, val_dataloader):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            print("Epoch", epoch + 1)
            self.model.train()
            train_loss_epoch = 0
            for batch_idx, batch in enumerate(train_dataloader):
                self.optimiser.zero_grad()
                loss = self.get_loss(batch)
                train_loss_epoch += loss.item()
                loss.backward()
                self.optimiser.step()
            train_loss_epoch /= len(train_dataloader.dataset)
            print("Training loss", train_loss_epoch)

            with torch.no_grad():
                self.model.eval()
                val_loss_epoch = 0
                for val_batch_idx, val_batch in enumerate(val_dataloader):
                    val_loss = self.get_loss(val_batch)
                    val_loss_epoch += val_loss.item()
                val_loss_epoch /= len(val_dataloader.dataset)
                print("Validation loss", val_loss_epoch)
                self.early_stopper.register_loss(val_loss_epoch)

            if self.early_stopper.should_stop():
                break

    def get_loss(self, batch):
        demonstrations = batch["demonstration"].to(self.device)
        # targets (b, dem_len, 6)
        targets = batch["demonstration_targets"].to(self.device)
        # lengths (b)
        lengths = batch["lengths"].to(self.device)
        max_len = lengths.max().item()
        (b,) = lengths.size()
        # predicted targets (b, dem_len, 6)
        predicted_targets, _hidden_state = self.model(demonstrations)
        ones = torch.ones(b, max_len)
        mask = (torch.arange(0, max_len) * ones).to(self.device) < lengths.unsqueeze(-1)
        mask = mask.float()
        # mask (b, dem_len)
        loss = self.loss(predicted_targets, targets)
        return torch.sum(loss * mask.unsqueeze(-1)) / torch.sum(mask)

    def get_best_info(self):
        return self.best_info

    def get_info(self):
        return dict(
            name=self.name,
            state_dict=self.model.state_dict(),
            hidden_size=self.hidden_size
        )
