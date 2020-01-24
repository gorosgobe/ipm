import torch


class BaselineNetwork(torch.nn.Module):
    def __init__(self, _image_width, _image_height):
        """
        Similar to FullImageNetwork, but takes only relative positions as target position estimate
        :param image_width: Unused
        :param image_height: Unused
        """
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=6, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        out_fc1 = torch.nn.functional.relu(self.fc1.forward(x))
        out_fc2 = torch.nn.functional.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3


class FullImageNetwork(torch.nn.Module):
    """
    Removes max pool layers, applying only stride, network for full image, predicting
    velocities and orientation change.
    """

    def __init__(self, image_width, image_height):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        self.fc1 = torch.nn.Linear(in_features=2240, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        batch_size = x.size()[0]
        out_conv1 = torch.nn.functional.relu(self.batch_norm1.forward(self.conv1.forward(x)))
        out_conv2 = torch.nn.functional.relu(self.batch_norm2.forward(self.conv2.forward(out_conv1)))
        out_conv3 = torch.nn.functional.relu(self.batch_norm3.forward(self.conv3.forward(out_conv2)))
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = torch.nn.functional.relu(self.fc1.forward(out_conv3))
        out_fc2 = torch.nn.functional.relu(self.fc2.forward(out_fc1))
        out_fc3 = self.fc3.forward(out_fc2)
        return out_fc3


class Network(torch.nn.Module):
    def __init__(self, image_width, image_height):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=1)
        self.batch_norm2 = torch.nn.BatchNorm2d(32)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=1)
        self.batch_norm3 = torch.nn.BatchNorm2d(16)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(in_features=240, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=3)

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
        out_fc2 = self.fc2.forward(out_fc1)
        return out_fc2
