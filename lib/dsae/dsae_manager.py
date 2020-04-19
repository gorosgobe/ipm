import torch

from lib.common.early_stopper import EarlyStopper
from lib.common.saveable import BestSaveable
from lib.dsae.dsae_networks import TargetVectorLoss, TargetDecoder
from lib.dsae.dsae_plot import plot_reconstruction_images, plot_full_demonstration


class DSAEManager(BestSaveable):
    def __init__(self, name, model, num_epochs, optimiser, device, criterion, criterion_params, add_g_slow, patience,
                 plot,
                 plot_params):
        super().__init__()
        self.name = name
        self.model = model
        self.num_epochs = num_epochs
        self.optimiser = optimiser
        self.device = device
        self.criterion = criterion
        self.criterion_params = criterion_params
        self.add_g_slow = add_g_slow
        self.plot = plot
        if self.plot and plot_params is None:
            raise ValueError("Need parameters and transforms for the plotting!")
        self.plot_params = plot_params

        self.early_stopper = EarlyStopper(patience=patience, saveable=self)
        self.patience = patience
        self.training_losses = []
        self.validation_losses = []
        self.best_info = self.get_info()

    def get_info(self):
        return dict(
            state_dict=self.model.state_dict(),
            optimiser_state_dict=self.optimiser.state_dict(),
            num_epochs=self.num_epochs,
            name=self.name,
            add_g_slow=self.add_g_slow,
            criterion=type(self.criterion),
            criterion_params=self.criterion_params,
            training_losses=self.training_losses,
            validation_losses=self.validation_losses
        )

    def get_best_info(self):
        return self.best_info

    def get_loss(self, batch):
        images = batch["images"].to(self.device)
        targets = batch["target_image"].to(self.device)
        target_vel_rot = batch["target_vel_rot"].to(self.device)
        # select image to reconstruct
        center_images = images[:, 1]

        predicted_vel_rot = None
        if isinstance(self.model.decoder, TargetDecoder):
            reconstructed, predicted_vel_rot = self.model(center_images)
        else:
            reconstructed = self.model(center_images)

        ft_minus1 = ft = ft_plus1 = None
        if self.add_g_slow:
            ft_minus1 = self.model.encoder(images[:, 0])
            ft = self.model.encoder(images[:, 1])
            ft_plus1 = self.model.encoder(images[:, 2])

        target_contrib = 0
        if isinstance(self.criterion, TargetVectorLoss):
            raw_recon_loss, raw_g_slow_contrib, raw_target_contrib = self.criterion(
                reconstructed=reconstructed,
                target=targets,
                ft_minus1=ft_minus1,
                ft=ft,
                ft_plus1=ft_plus1,
                pred_vel_rot=predicted_vel_rot,
                target_vel_rot=target_vel_rot
            )

            # 0.01
            recon_loss = self.criterion_params[0] * raw_recon_loss
            g_slow_contrib = self.criterion_params[1] * raw_g_slow_contrib
            target_contrib = self.criterion_params[2] * raw_target_contrib

            loss = (recon_loss + g_slow_contrib + target_contrib) / len(images)
        else:
            recon_loss, g_slow_contrib = self.criterion(
                reconstructed=reconstructed, target=targets, ft_minus1=ft_minus1, ft=ft, ft_plus1=ft_plus1
            )
            loss = (recon_loss + g_slow_contrib) / len(images)

        return loss, recon_loss, g_slow_contrib, target_contrib

    def train(self, train_dataloader, validation_dataloader):
        for epoch in range(self.num_epochs):
            loss_epoch = 0
            recon_loss_epoch = 0
            g_slow_contrib_epoch = 0
            target_vel_rot_epoch = 0
            self.model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                self.optimiser.zero_grad()
                loss, recon_loss, g_slow_contrib, target_contrib = self.get_loss(batch)
                # log losses
                loss_epoch += loss.item()
                recon_loss_epoch += recon_loss.item()
                g_slow_contrib_epoch += g_slow_contrib.item()
                target_vel_rot_epoch += target_contrib.item()

                loss.backward()
                self.optimiser.step()

            print((
                f"Epoch {epoch + 1}: {loss_epoch / len(train_dataloader)}, Recon loss "
                f"{recon_loss_epoch / len(train_dataloader)}, g_slow loss {g_slow_contrib_epoch / len(train_dataloader)} "
                f"Target vel rot loss {target_vel_rot_epoch / len(train_dataloader)}"
            ))
            self.training_losses.append(loss_epoch / len(train_dataloader))

            # validate after every epoch
            self.model.eval()
            with torch.no_grad():
                val_loss_epoch = 0
                for val_batch_idx, val_batch in enumerate(validation_dataloader):
                    val_loss, _, _, _ = self.get_loss(val_batch)
                    val_loss_epoch += val_loss.item()

                complete_val_loss = val_loss_epoch / len(validation_dataloader)
                print("Validation loss:", complete_val_loss)
                self.early_stopper.register_loss(complete_val_loss)
                self.validation_losses.append(complete_val_loss)

                if self.plot:
                    if epoch % 5 == 0:
                        plot_reconstruction_images(
                            epoch, self.name, self.plot_params["dataset"], self.model, self.model.decoder,
                            self.plot_params["upsample_transform"],
                            self.plot_params["grayscale"], self.device
                        )
                    if epoch % 20 == 0:
                        plot_full_demonstration(
                            epoch, self.name, self.plot_params["dataset"], self.model,
                            self.plot_params["grayscale"], self.device, self.plot_params["latent_dimension"]
                        )

            if self.early_stopper.should_stop():
                print(f"Stopping, patience {self.patience} reached...")
                if self.plot:
                    # we want to plot what our best model would have outputted
                    # load best weights
                    self.model.load_state_dict(self.get_best_info()["state_dict"])
                    # and now plot
                    plot_reconstruction_images(
                        epoch, self.name, self.plot_params["dataset"], self.model, self.model.decoder,
                        self.plot_params["upsample_transform"],
                        self.plot_params["grayscale"], self.device
                    )
                    plot_full_demonstration(
                        epoch, self.name, self.plot_params["dataset"], self.model,
                        self.plot_params["grayscale"], self.device, self.plot_params["latent_dimension"]
                    )
                break
