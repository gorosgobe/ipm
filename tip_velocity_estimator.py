import os
import time

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import EarlyStopping
from ignite.metrics import Loss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class ImageTipVelocitiesDataset(torch.utils.data.Dataset):
    def __init__(self, csv, root_dir, resize=None, transform=None):
        self.tip_velocities_frame = pd.read_csv(csv) 
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.tip_velocities_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.tip_velocities_frame.iloc[idx, 0])
        image = imageio.imread(img_name)

        tip_velocities = self.tip_velocities_frame.iloc[idx, 1:]
        tip_velocities = np.array(tip_velocities, dtype=np.float32)

        sample = {'image': image, 'tip_velocities': tip_velocities}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

class Network(torch.nn.Module):
    def __init__(self, image_width, image_height):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_size = 20 * int(image_width * image_height / (8**2))
        self.fc1 = torch.nn.Linear(in_features=self.fc_size, out_features=100)
        self.fc2 = torch.nn.Linear(in_features=100, out_features=3)

    def forward(self, x):
        out_conv1 = self.conv1.forward(torch.nn.functional.relu(x))
        out_maxpool1 = self.pool1.forward(out_conv1)

        out_conv2 = self.conv2.forward(torch.nn.functional.relu(out_maxpool1))
        out_maxpool2 = self.pool2.forward(out_conv2)

        out_conv3 = self.conv3.forward(torch.nn.functional.relu(out_maxpool2))
        out_maxpool3 = self.pool3.forward(out_conv3)
        out_maxpool3 = out_maxpool3.view(-1, self.fc_size)
        out_fc1 = self.fc1.forward(torch.nn.functional.relu(out_maxpool3))
        out_fc2 = self.fc2.forward(torch.nn.functional.relu(out_fc1))

        return out_fc2

class TipVelocityEstimator(object):
    def __init__(self, batch_size, learning_rate, image_size, transforms=None):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.image_size = image_size
        width, height = self.image_size
        self.network = Network(width, height)
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.MSELoss()
        self.trainer = self._create_trainer()
        self.evaluator = self._create_evaluator()
        # set in train()
        self.train_data_loader = None
        self.val_loader = None
        self.training_losses = []
        self.validation_losses = []
        self.resize_transform = ResizeTransform(self.image_size)
        # transformations applied to input, except for initial resize
        self.transforms = transforms

    def train(self, data_loader, max_epochs, validate_epochs=10, val_loader=None):
        """
        validate_epochs: After how many epochs do we want to 
        validate our model against the validation dataset
        """
        self.train_data_loader = data_loader
        self.validate_epochs   = validate_epochs
        self.val_loader        = val_loader
        return self.trainer.run(data_loader, max_epochs)

    def get_network(self):
        return self.network

    @staticmethod
    def prepare_batch(batch, device, non_blocking):
        return batch["image"], batch["tip_velocities"]

    def epoch_started(self):
        def static_epoch_started(trainer):
            print("Epoch {}".format(trainer.state.epoch))

        return static_epoch_started

    def epoch_completed(self):
        def static_epoch_completed(trainer):
            self.evaluator.run(self.train_data_loader)
            metrics = self.evaluator.state.metrics
            loss = metrics["loss"]
            self.training_losses.append((trainer.state.epoch, loss))
            print("Epoch {}, training loss {}".format(trainer.state.epoch, loss))

        return static_epoch_completed

    def epoch_validate(self):
        def static_epoch_validate(trainer):
            if self.val_loader is not None and trainer.state.epoch % self.validate_epochs == 0:
                # validate against validation dataset
                self.evaluator.run(self.val_loader)
                metrics = self.evaluator.state.metrics
                loss = metrics["loss"]
                self.validation_losses.append((trainer.state.epoch, loss))
                print("Validation loss for epoch {}: {}".format(trainer.state.epoch, loss))

        return static_epoch_validate

    def _create_trainer(self):
        trainer = create_supervised_trainer(
            self.network, 
            self.optimiser, 
            self.loss_func,
            prepare_batch=TipVelocityEstimator.prepare_batch
        )
        trainer.add_event_handler(Events.EPOCH_STARTED, self.epoch_started())
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed())
        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_validate())
        return trainer

    def _create_evaluator(self, early_stopping=True, patience=10):
        evaluator = create_supervised_evaluator(self.network, metrics={
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

    def save(self, path, epoch=-1):
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "epoch": epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "transforms": self.transforms,
            "image_size": self.image_size
        }, path)

    def load_parameters(self, state_dict):
        self.network.load_state_dict(state_dict)

    def load_optimiser_parameters(self, state_dict):
        self.optimiser.load_state_dict(state_dict)

    @staticmethod
    def load(path):
        info = torch.load(path)
        batch_size    = info["batch_size"]
        learning_rate = info["learning_rate"]
        state_dict    = info["model_state_dict"]
        optimiser_state_dict = info["optimiser_state_dict"]
        transforms    = info["transforms"]
        image_size    = info["image_size"]

        estimator = TipVelocityEstimator(batch_size, learning_rate, image_size, transforms)
        estimator.load_parameters(state_dict)
        estimator.load_optimiser_parameters(optimiser_state_dict)

        return estimator

    def resize_image(self, image):
        """
        Resizes image to size of image controller was trained with
        """
        return self.resize_transform(image)

    def predict(self, batch):
        return self.network.forward(batch)

class ResizeTransform(object):
    def __init__(self, size):
        self.size = size # tuple, like (128, 96)

    def __call__(self, image):
        return cv2.resize(image, dsize=self.size)

if __name__ == "__main__":

    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    size = (128, 96)
    preprocessing_transforms = torchvision.transforms.Compose([
        ResizeTransform(size), 
        transforms
    ])

    dataset = ImageTipVelocitiesDataset(
        csv="./datadummyvelocity/velocities.csv", 
        root_dir="./datadummyvelocity",
        transform=preprocessing_transforms
    )

    batch_size = 50

    n_training_samples = int(0.9 * len(dataset))
    n_test_samples     = len(dataset) - n_training_samples

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_training_samples, n_test_samples])
    train_data_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data_loader   = DataLoader(test_dataset, batch_size=4, shuffle=True)

    tip_velocity_estimator = TipVelocityEstimator(
        batch_size=batch_size,
        learning_rate=0.0001,
        image_size=(128, 96),
        # transforms without initial resize, so they can be pickled correctly
        transforms=transforms
    
    )
    tip_velocity_estimator.train(
        train_data_loader, 
        max_epochs=30, # or stop early with patience 5
        validate_epochs=1, 
        val_loader=test_data_loader
    )
    
    training_epochs, training_losses = zip(*tip_velocity_estimator.training_losses)
    plt.plot(training_epochs, training_losses, label="Train loss")
    validation_epochs, validation_losses = zip(*tip_velocity_estimator.validation_losses)
    plt.plot(validation_epochs, validation_losses, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.show()
    # save the model
    t = int(time.time())
    tip_velocity_estimator.save("models/model{}.pt".format(t))
