from collections import OrderedDict

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from lib.common.early_stopper import EarlyStopper
from lib.common.saveable import BestSaveable


class STNManager(BestSaveable):
    def __init__(self, name, stn, device, loc_lr=1e-4, model_lr=1e-2):
        super().__init__()
        self.name = name
        self.stn = stn
        self.best_info = None
        self.device = device
        self.loss = nn.MSELoss()
        self.stn_optimiser = torch.optim.SGD(self.stn.localisation_param_regressor.parameters(), lr=loc_lr,
                                             momentum=0.9, weight_decay=1e-4)
        self.model_optimiser = torch.optim.SGD(self.stn.model.parameters(), lr=model_lr)
        self.loc_lr = loc_lr
        self.model_lr = model_lr
        self.stn_scheduler = ReduceLROnPlateau(self.stn_optimiser, factor=0.5, patience=5, threshold=1e-5)
        self.model_scheduler = ReduceLROnPlateau(self.model_optimiser, factor=0.5, patience=5, threshold=1e-5)
        self.early_stopper = EarlyStopper(patience=15, saveable=self)

    def get_loss(self, batch, params=None):
        images = batch["image"].to(self.device)
        tip_velocities = batch["tip_velocities"].to(self.device)
        rotations = batch["rotations"].to(self.device)
        targets = torch.cat((tip_velocities, rotations), 1)
        predicted_targets = self.stn(images, params=params)
        return self.loss(predicted_targets, targets)

    def get_dsae_guide_init_loss(self, batch, dsae_init_index):
        images = batch["image"].to(self.device)
        self.stn.localisation_param_regressor.dsae.eval()  # just to make sure
        with torch.no_grad():
            # spatial features (B, C, 2)
            # dsae does not take coordconv map
            spatial_features = self.stn.localisation_param_regressor.dsae(images[:, :3])
            # target translation should be the indexed spatial feature, for initialisation
            # (B, 2)
            target_indexed_spatial_feature = spatial_features[:, dsae_init_index]

        # (s, t_x, t_y) : (B, 3) if scale is not None, (B, 2) otherwise
        regression_transform_params = self.stn.localisation_param_regressor.fc_model(spatial_features)
        b, transform_size = regression_transform_params.size()
        # use only t_x, and t_y
        only_translation_idx = 1 if transform_size == 3 else 0
        return self.loss(regression_transform_params[:, only_translation_idx:], target_indexed_spatial_feature)


    @staticmethod
    def pseudo_infinite_sampling(dataloader, max_count):
        count = 0
        while True and count < max_count:
            for idx, batch in enumerate(dataloader):
                yield batch
                count += 1

    def train(self, num_epochs, train_dataloader, val_dataloader, test_dataloader, pre_training=True,
              double_meta_loss=False, dsae_guide_init_params=None):
        self.stn.to(self.device)
        if dsae_guide_init_params is not None:
            dsae_init_epochs = dsae_guide_init_params["dsae_init_epochs"]
            dsae_init_index = dsae_guide_init_params["dsae_init_index"]
            self.stn.eval()
            # only train param regressor layer, not the DSAE
            self.stn.localisation_param_regressor.fc_model.train()
            regressor_optimiser = torch.optim.Adam(self.stn.localisation_param_regressor.fc_model.parameters(), lr=0.001)
            print("Starting initialisation regression...")
            for epoch in range(dsae_init_epochs):
                print("Guide init epoch", epoch + 1)
                train_dsae_guide_init_epoch = 0
                for batch_idx, batch in enumerate(train_dataloader):
                    regressor_optimiser.zero_grad()
                    loss = self.get_dsae_guide_init_loss(batch, dsae_init_index)
                    train_dsae_guide_init_epoch += loss.item()
                    loss.backward()
                    regressor_optimiser.step()
                print("Guide init loss", train_dsae_guide_init_epoch / len(train_dataloader.dataset))

        print("Starting meta learning...")
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            self.stn.localisation_param_regressor.train()
            self.stn.model.train()
            # one epoch of training for regression model with default transformation (set to identity at the beginning)
            if pre_training:
                train_loss_epoch = 0
                for batch_idx, batch in enumerate(train_dataloader):
                    self.stn_optimiser.zero_grad()
                    self.model_optimiser.zero_grad()
                    loss = self.get_loss(batch)
                    train_loss_epoch += loss.item()
                    loss.backward()
                    self.model_optimiser.step()
                print("Train loss", train_loss_epoch / len(train_dataloader.dataset))

            train_val_train_loss_epoch = 0
            # find parameters theta prime from training data
            for train_batch, val_batch in zip(
                    self.pseudo_infinite_sampling(train_dataloader, len(train_dataloader)),
                    self.pseudo_infinite_sampling(val_dataloader, len(train_dataloader))
            ):
                # we now have training and validation batches
                self.stn_optimiser.zero_grad()
                self.model_optimiser.zero_grad()
                train_loss = self.get_loss(train_batch)
                train_val_train_loss_epoch += train_loss.item()

                # theta
                fast_weights = OrderedDict((name, param) for (name, param) in self.stn.model.named_parameters())
                # gradient with respect to regression network, retain it in the graph
                grads = torch.autograd.grad(train_loss, self.stn.model.parameters(), create_graph=True)
                # theta prime
                fast_weights = OrderedDict(
                    (name, param - self.model_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))

                # compute val loss using theta prime
                val_loss = self.get_loss(val_batch, params=fast_weights)
                if double_meta_loss:
                    val_loss += self.get_loss(train_batch, params=fast_weights)
                val_loss.backward()
                self.stn_optimiser.step()

                if not pre_training:
                    # update theta prime
                    self.model_optimiser.step()

            train_val_train_loss_epoch /= len(train_dataloader.dataset)
            print("Train val loss", train_val_train_loss_epoch)

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
                avg_test_eval_loss_epoch = test_eval_loss_epoch / len(test_dataloader.dataset)

                self.stn_scheduler.step(avg_test_eval_loss_epoch)
                self.model_scheduler.step(avg_test_eval_loss_epoch)

                self.early_stopper.register_loss(avg_test_eval_loss_epoch)
                print("Test EVAL:", avg_test_eval_loss_epoch)
                if self.early_stopper.should_stop():
                    print("Patience reached, stopping...")
                    break

    def retrain(self, num_epochs, train_dataloader, val_dataloader, test_dataloader):
        # clear best info, restart early stopper
        self.best_info = None
        self.early_stopper = EarlyStopper(patience=10, saveable=self)
        optimiser = torch.optim.Adam(self.stn.parameters(), lr=0.0001)

        for epoch in range(num_epochs):
            print("Retraining epoch", epoch + 1)
            self.stn.train()
            train_loss_epoch = 0
            for batch_idx, batch in enumerate(train_dataloader):
                optimiser.zero_grad()
                loss = self.get_loss(batch)
                train_loss_epoch += loss.item()
                loss.backward()
                optimiser.step()
            train_loss_epoch /= len(train_dataloader.dataset)
            print("Training loss", train_loss_epoch)

            self.stn.eval()
            val_loss_epoch = 0
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                optimiser.zero_grad()
                val_loss = self.get_loss(val_batch)
                val_loss_epoch += val_loss.item()
            val_loss_epoch /= len(val_dataloader.dataset)
            print("Validation loss", val_loss_epoch)

            test_loss_epoch = 0
            for test_batch_idx, test_batch in enumerate(test_dataloader):
                optimiser.zero_grad()
                test_loss = self.get_loss(test_batch)
                test_loss_epoch += test_loss.item()
            test_loss_epoch /= len(test_dataloader.dataset)
            print("Test loss", test_loss_epoch)

            self.early_stopper.register_loss(val_loss_epoch)
            if self.early_stopper.should_stop():
                break

    def get_info(self):
        return dict(
            name=self.name,
            stn_state_dict=self.stn.state_dict()
        )

    def get_best_info(self):
        return self.best_info
