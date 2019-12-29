import json
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
from torch.utils.data import DataLoader, Subset


class ImageTipVelocitiesDataset(torch.utils.data.Dataset):
    def __init__(self, csv, metadata, root_dir, resize=None, transform=None):
        self.tip_velocities_frame = pd.read_csv(csv, header=None)
        self.root_dir = root_dir
        self.transform = transform
        with open(metadata, "r") as m:
            metadata_content = m.read()
        self.demonstration_metadata = json.loads(metadata_content)

    def get_indices_for_demonstration(self, d_idx):
        demonstration_data = self.demonstration_metadata["demonstrations"][str(d_idx)]
        return demonstration_data["start"], demonstration_data["end"]

    def get_num_demonstrations(self):
        return self.demonstration_metadata["num_demonstrations"]

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
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(10)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=15, kernel_size=5, stride=1, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(15)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(20)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(in_features=120, out_features=30)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(in_features=30, out_features=3)


    def forward(self, x):
        batch_size = x.size()[0]
        out_conv1 = torch.nn.functional.relu(self.batch_norm1.forward(self.conv1.forward(x)))
        out_conv1 = self.pool1.forward(out_conv1)
        out_conv2 = torch.nn.functional.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        out_conv2 = self.pool2.forward(out_conv2)
        out_conv3 = torch.nn.functional.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = self.pool3.forward(out_conv3)
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = torch.nn.functional.relu(self.fc1.forward(out_conv3))
        out_fc2 = self.fc2.forward(self.dropout.forward(out_fc1))
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
        self.training_evaluator = self._create_evaluator()
        self.validation_evaluator = self._create_evaluator(early_stopping=True, patience=10)
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
        self.validate_epochs = validate_epochs
        self.val_loader = val_loader
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
            self.training_evaluator.run(self.train_data_loader)
            metrics = self.training_evaluator.state.metrics
            loss = metrics["loss"]
            self.training_losses.append((trainer.state.epoch, loss))
            print("Training loss {}".format(loss))

        return static_epoch_completed

    def epoch_validate(self):
        def static_epoch_validate(trainer):
            if self.val_loader is not None and trainer.state.epoch % self.validate_epochs == 0:
                # validate against validation dataset
                self.validation_evaluator.run(self.val_loader)
                metrics = self.validation_evaluator.state.metrics
                loss = metrics["loss"]
                self.validation_losses.append((trainer.state.epoch, loss))
                print("Validation loss: {}".format(loss))

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

    def _create_evaluator(self, early_stopping=False, patience=10):
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
        batch_size = info["batch_size"]
        learning_rate = info["learning_rate"]
        state_dict = info["model_state_dict"]
        optimiser_state_dict = info["optimiser_state_dict"]
        transforms = info["transforms"]
        image_size = info["image_size"]

        estimator = TipVelocityEstimator(batch_size, learning_rate, image_size, transforms)
        estimator.load_parameters(state_dict)
        estimator.load_optimiser_parameters(optimiser_state_dict)

        return estimator

    def resize_image(self, image):
        """
        Resize image to size of image that the controller was trained with
        """
        return self.resize_transform(image)

    def predict(self, batch):
        with torch.no_grad():
            return self.network.forward(batch)


class ResizeTransform(object):
    def __init__(self, size):
        self.size = size  # tuple, like (128, 96)

    def __call__(self, image):
        return cv2.resize(image, dsize=self.size)


if __name__ == "__main__":
    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    size = (64, 48)
    preprocessing_transforms = torchvision.transforms.Compose([\
        ResizeTransform(size),
        transforms
    ])

    dataset = ImageTipVelocitiesDataset(
        csv="./croppeddatalimitedshift30/velocities.csv",
        metadata="./croppeddatalimitedshift30/metadata.json",
        root_dir="./croppeddatalimitedshift30",
        transform=preprocessing_transforms
    )

    batch_size = 32

    total_demonstrations = dataset.get_num_demonstrations()
    n_training_demonstrations = int(0.9 * total_demonstrations)
    start_train, _ = dataset.get_indices_for_demonstration(0)
    _, end_train = dataset.get_indices_for_demonstration(n_training_demonstrations - 1)
    training_demonstrations = Subset(dataset, np.arange(start_train, end_train + 1))

    n_test_demonstrations = total_demonstrations - n_training_demonstrations
    start_test, _ = dataset.get_indices_for_demonstration(n_training_demonstrations)
    _, end_test = dataset.get_indices_for_demonstration(n_training_demonstrations + n_test_demonstrations - 1)
    test_demonstrations = Subset(dataset, np.arange(start_test, end_test + 1))

    print("Training demonstrations: ", n_training_demonstrations)
    print("Test demonstrations: ", n_test_demonstrations)

    train_data_loader = DataLoader(training_demonstrations, batch_size=batch_size, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_demonstrations, batch_size=4, shuffle=True)

    tip_velocity_estimator = TipVelocityEstimator(
        batch_size=batch_size,
        learning_rate=0.0001,
        image_size=size,
        # transforms without initial resize, so they can be pickled correctly
        transforms=transforms

    )
    tip_velocity_estimator.train(
        train_data_loader,
        max_epochs=100,  # or stop early with patience 10
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
