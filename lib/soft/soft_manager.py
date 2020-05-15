import torch
from torch import nn, distributions

from lib.common.early_stopper import EarlyStopper
from lib.common.saveable import BestSaveable
from lib.soft.soft import SoftCNNLSTMNetwork


class SoftManager(BestSaveable):

    def __init__(self, name, dataset, device, hidden_size, is_coord, entropy_lambda=1.0):
        super().__init__()
        self.name = name
        self.model = SoftCNNLSTMNetwork(hidden_size=hidden_size, is_coord=is_coord)
        self.is_coord = is_coord
        self.dataset = dataset
        self.device = device
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss = nn.MSELoss(reduction="none")
        self.early_stopper = EarlyStopper(patience=10, saveable=self)
        self.best_info = None
        self.hidden_size = hidden_size
        self.entropy_lambda = entropy_lambda

    def train(self, num_epochs, train_dataloader, val_dataloader):
        self.model.to(self.device)
        for epoch in range(num_epochs):
            print("Epoch", epoch + 1)
            self.model.train()
            train_loss_epoch = 0
            train_target_epoch = 0
            train_entropy_epoch = 0
            for batch_idx, batch in enumerate(train_dataloader):
                self.optimiser.zero_grad()
                target_contribution, entropy_contribution, mask_total = self.get_loss(batch)
                train_target_epoch += target_contribution / mask_total
                train_entropy_epoch += entropy_contribution / mask_total

                loss = (target_contribution + entropy_contribution) / mask_total
                train_loss_epoch += loss.item()
                loss.backward()

                self.optimiser.step()

            train_loss_epoch /= len(train_dataloader.dataset)
            train_target_epoch /= len(train_dataloader.dataset)
            train_entropy_epoch /= len(train_dataloader.dataset)
            print(f"Training loss {train_loss_epoch}, target {train_target_epoch}, entropy {train_entropy_epoch}")

            with torch.no_grad():
                self.model.eval()
                val_loss_epoch = 0
                val_target_epoch = 0
                val_entropy_epoch = 0
                for val_batch_idx, val_batch in enumerate(val_dataloader):
                    val_target_contribution, val_entropy_contribution, val_mask_total = self.get_loss(val_batch)
                    val_target_epoch += val_target_contribution / val_mask_total
                    val_entropy_epoch += val_entropy_contribution / val_mask_total

                    val_loss = (val_target_contribution + val_entropy_contribution) / val_mask_total
                    val_loss_epoch += val_loss.item()

                val_loss_epoch /= len(val_dataloader.dataset)
                val_target_epoch /= len(val_dataloader.dataset)
                val_entropy_epoch /= len(val_dataloader.dataset)
                print(f"Validation loss {val_loss_epoch}, target {val_target_epoch}, entropy {val_entropy_epoch}")
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
        predicted_targets, _hidden_state, importances = self.model(demonstrations)
        ones = torch.ones(b, max_len)
        mask = (torch.arange(0, max_len) * ones).to(self.device) < lengths.unsqueeze(-1)
        mask = mask.float()
        # mask (b, dem_len)
        # importance_stack (b, dem_len, 1, H'xW')
        importance_stack = torch.stack(importances).transpose(0, 1)
        entropies = distributions.Categorical(probs=importance_stack).entropy()
        # entropies (b, dem_len, 1)
        entropy_contribution = torch.sum(entropies * mask.unsqueeze(-1))
        loss = self.loss(predicted_targets, targets)
        target_contribution = torch.sum(loss * mask.unsqueeze(-1))
        mask_total = torch.sum(mask)
        return target_contribution, self.entropy_lambda * entropy_contribution, mask_total

    def get_best_info(self):
        return self.best_info

    def get_info(self):
        return dict(
            name=self.name,
            state_dict=self.model.state_dict(),
            hidden_size=self.hidden_size,
            is_coord=self.is_coord
        )
