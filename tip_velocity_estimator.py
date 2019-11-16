import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision
import pandas as pd
import os
import imageio

class ImageTipVelocitiesDataset(torch.utils.data.Dataset):
    def __init__(self, csv, root_dir, transform=None):
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
        tip_velocities = np.array(tip_velocities)

        sample = {'image': image, 'tip_velocities': tip_velocities}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

class Network(torch.nn.Module):
    def __init__(self, image_width, image_height):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=18, kernel_size=3, stride=1, padding=0)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=18, out_channels=40, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv2layer = torch.nn.Conv2d(in_channels=18, out_channels=50, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(in_features=int(image_width * image_height / 8), out_features=100)
        self.fc2 = torch.nn.Linear(in_features=100, out_features=3)

    def forward(self, x):
        out_conv1 = self.conv1.forward(torch.nn.functional.relu(x))
        print(out_conv1.size())
        out_maxpool1 = self.pool1.forward(out_conv1)

        out_conv2 = self.conv2.forward(torch.nn.functional.relu(out_maxpool1))
        print(out_conv2.size())
        out_maxpool2 = self.pool2.forward(out_conv2)

        out_conv3 = self.conv3.forward(torch.nn.functional.relu(out_maxpool2))
        print(out_conv3.size())
        out_maxpool3 = self.pool3.forward(out_conv3)
        print(out_maxpool3.size())
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

    preprocessing_transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    dataset = ImageTipVelocitiesDataset(csv="./data/velocities.csv", root_dir="./data", transform=preprocessing_transforms)
    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample["image"].shape, sample["tip_velocities"].shape)

    batch_size = 4

    n_training_samples = int(0.8 * len(dataset))
    train_sampler      = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

    n_test_samples     = int(0.2 * len(dataset))
    test_sampler       = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

    train_data_loader  = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_data_loader   = DataLoader(dataset, batch_size=4, sampler=test_sampler)

    tip_velocity_estimator = TipVelocityEstimator(batch_size=batch_size, learning_rate=0.0001)

    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_index, (images, velocities) in enumerate(train_data_loader):

            loss = tip_velocity_estimator.train(images, velocities)
            print(loss)