import cv2
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision
import pandas as pd
import os
import imageio
import matplotlib.pyplot as plt

class ImageTipVelocitiesDataset(torch.utils.data.Dataset):
    def __init__(self, csv, root_dir, resize=None, transform=None):
        self.tip_velocities_frame = pd.read_csv(csv) 
        self.root_dir = root_dir
        self.transform = transform
        # if not None, size we want to resize each image to
        self.resize = resize

    def __len__(self):
        return len(self.tip_velocities_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.tip_velocities_frame.iloc[idx, 0])
        image = imageio.imread(img_name)
        if self.resize:
            image = cv2.resize(image, dsize=self.resize)

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
        self.pool1 = torch.nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=10, out_channels=15, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_size = 20 * int(image_width * image_height / (16**2))
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
    def __init__(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.network = Network(128, 96)
        self.optimiser = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.MSELoss()
        
    def train(self, batch, velocities):
        self.optimiser.zero_grad()
        output = self.network.forward(batch)
        loss = self.loss_func(output, velocities)
        loss.backward()
        self.optimiser.step()
        return loss

    def predict(self, batch):
        return self.network.forward(batch)

if __name__ == "__main__":

    seed = 2019
    np.random.seed(seed)
    torch.manual_seed(seed)

    preprocessing_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageTipVelocitiesDataset(
        csv="./data/velocities.csv", 
        root_dir="./data", 
        resize=(128, 96), 
        transform=preprocessing_transforms
    )

    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     print(i, sample["image"].shape, sample["tip_velocities"].shape)

    batch_size = 20
    training_ratio     = 0.9
    #validation_ratio   = (1 - training_ratio) / 2
    test_ratio         = 1 - training_ratio

    n_training_samples = int(0.8 * len(dataset))
    train_sampler      = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

    n_test_samples     = int(0.2 * len(dataset))
    test_sampler       = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

    train_data_loader  = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    test_data_loader   = DataLoader(dataset, batch_size=1, sampler=test_sampler)

    tip_velocity_estimator = TipVelocityEstimator(batch_size=batch_size, learning_rate=0.0001)

    losses = []
    # TODO: actually validation losses, need to add validation set
    test_losses = []
    num_epochs = 150
    validation_checkpoint = 10

    for epoch in range(1, num_epochs + 1):
        loss_total = 0.0
        count = 0
        for batch_index, sampled_batch in enumerate(train_data_loader):
            images = sampled_batch["image"]
            velocities = sampled_batch["tip_velocities"]
            loss = tip_velocity_estimator.train(images, velocities)
            loss_total += loss.item()
            count += 1

        avg_loss_epoch = loss_total / count
        losses.append(avg_loss_epoch)
        print("Epoch {}: train loss {}".format(epoch, avg_loss_epoch))

        if epoch % validation_checkpoint == 0:
            total_test_loss = 0.0
            count = 0
            for _, batch in enumerate(test_data_loader):
                images = batch["image"]
                velocities = batch["tip_velocities"]
                #print(velocities)
                #print("----" * 6)
                predicted_velocities = tip_velocity_estimator.predict(images)
                #print(predicted_velocities)
                total_test_loss += tip_velocity_estimator.loss_func(predicted_velocities, velocities)
                count += 1

            avg_test_loss = total_test_loss / count
            print("Prediction loss: {}".format(avg_test_loss))
            test_losses.append((epoch, avg_test_loss))

    plt.plot(range(1, num_epochs + 1), losses, label="Train loss")
    epochs, test_loss_values = zip(*test_losses)
    plt.plot(epochs, test_loss_values, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.show()