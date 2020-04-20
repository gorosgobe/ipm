import torch
from torch import nn, distributions

from lib.dsae.dsae_networks import SoftTarget
from lib.dsae.dsae_plot import plot_reconstruction_images
from lib.common.early_stopper import EarlyStopper
from lib.common.saveable import BestSaveable


class SoftSpatialDiscriminator(nn.Module, SoftTarget):
    """
    Learns weights that attend over spatial features, to try and obtain a single point representation that can
    still accurately predict the action
    """

    def __init__(self, latent_dimension):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.weights = nn.Parameter(torch.randn(latent_dimension // 2), requires_grad=True)
        self.attention_weights = None
        self.attended_location = None

        self.fc1 = torch.nn.Linear(in_features=2, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=6)
        self.activ = nn.ReLU()

    def forward(self, x):
        # x is of form (B, C, 2), contains spatial features in [-1, 1] x [-1, 1]
        b, c, _2 = x.size()
        assert c == self.latent_dimension // 2 and _2 == 2
        # weight (C,)
        # store attention weights so we can apply entropy penalty on it
        # (C, ) -> soft(1, C)
        self.attention_weights = nn.functional.softmax(self.weights, dim=0).unsqueeze(0)
        # soft(1, C, 2); (B, C, 2) * (1, C, 2) -> (B, C, 2) -sum-> (B, 2)
        self.attended_location = torch.sum(x * self.attention_weights.unsqueeze(-1).repeat(1, 1, 2), dim=1)
        # out (B, 6)
        out_fc1 = self.activ(self.fc1(self.attended_location))
        out_fc2 = self.fc2(out_fc1)
        return out_fc2


class SoftDiscriminatorLoss(object):
    def __init__(self):
        self.mse = nn.MSELoss(reduction="sum")

    def __call__(self, predicted, targets, attention_distributions):
        target_loss = self.mse(predicted, targets)
        entropies = distributions.Categorical(probs=attention_distributions).entropy()
        return target_loss, entropies


class DiscriminatorFeatureProvider(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        # make sure weights are frozen
        self.model.eval()
        with torch.no_grad():
            # returns (B, C*2 = latent dimension)
            return self.model.encoder(x)


class DiscriminatorManager(BestSaveable):
    def __init__(self, name, feature_provider, model, num_epochs, optimiser, loss_params, device, patience,
                 plot_params=None, plot=True):
        super().__init__()
        self.name = name
        # discriminator feature provider
        self.feature_provider = feature_provider
        self.model = model
        self.num_epochs = num_epochs
        self.optimiser = optimiser
        self.device = device
        self.loss = SoftDiscriminatorLoss()
        self.loss_params = loss_params
        self.patience = patience
        self.early_stopper = EarlyStopper(patience=self.patience, saveable=self)
        self.plot = plot
        self.plot_params = plot_params
        if self.plot and self.plot_params is None:
            raise ValueError("Need plotting parameters to plot... !")

        self.best_info = self.get_info()

    def get_best_info(self):
        return self.best_info

    def get_info(self):
        return dict(
            state_dict=self.model.state_dict(),
            optimiser_state_dict=self.optimiser.state_dict(),
            num_epochs=self.num_epochs,
            name=self.name,
            criterion_params=self.loss_params,
        )

    def get_loss(self, batch):
        targets = batch["target_vel_rot"].to(self.device)
        center_images = batch["images"][:, 1].to(self.device)

        features = self.feature_provider(center_images)
        predicted_targets = self.model(features)
        target_loss, entropies = self.loss(predicted=predicted_targets, targets=targets,
                                           attention_distributions=self.model.attention_weights)
        target_contrib = self.loss_params[0] * target_loss
        entropy_contrib = self.loss_params[1] * entropies
        # average loss per sample in batch
        # entropy is 1 as we dont have batch-sized attention weights
        loss = (target_contrib / len(batch)) + entropy_contrib
        return loss, target_contrib, entropy_contrib

    def train(self, train_dataloader, validation_dataloader):
        for epoch in range(self.num_epochs):
            loss_epoch = 0
            target_loss_epoch = 0
            entropy_epoch = 0
            self.model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                self.optimiser.zero_grad()
                loss, target_contrib, entropy_contrib = self.get_loss(batch)
                loss_epoch += loss.item()
                target_loss_epoch += target_contrib.item()
                entropy_epoch += entropy_contrib.item()

                loss.backward()
                self.optimiser.step()

            print(
                f"Epoch {epoch + 1}: Training loss {loss_epoch / len(train_dataloader)}, "
                f"target loss {target_loss_epoch / len(train_dataloader)}, "
                f"entropy {entropy_epoch / len(train_dataloader)}"
            )

            self.model.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                for val_batch_idx, val_batch in enumerate(validation_dataloader):
                    val_loss, _, _ = self.get_loss(val_batch)
                    val_loss_epoch += val_loss.item()

                complete_val_loss = val_loss_epoch / len(validation_dataloader)
                print("Validation loss", complete_val_loss)
                self.early_stopper.register_loss(complete_val_loss)

                if self.plot:
                    if epoch % 20 == 0:
                        plot_reconstruction_images(
                            epoch=epoch, name=self.name, dataset=self.plot_params["dataset"], model=self.plot_params["feature_model"],
                            attender=self.model,  # attender is discriminator
                            upsample_transform=self.plot_params["upsample_transform"],
                            grayscale=self.plot_params["grayscale"], device=self.device, attender_discriminator=True
                        )

            if self.early_stopper.should_stop():
                print(f"Stopping, patience {self.patience} reached.")
                break

        if self.plot:
            # load best weights
            self.model.load_state_dict(self.get_best_info()["state_dict"])
            plot_reconstruction_images(
                epoch="final", name=self.name, dataset=self.plot_params["dataset"], model=self.plot_params["feature_model"],
                attender=self.model,  # attender is discriminator
                upsample_transform=self.plot_params["upsample_transform"],
                grayscale=self.plot_params["grayscale"], device=self.device, attender_discriminator=True
            )
