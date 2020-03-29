from torch import nn
from torchmeta.modules import MetaModule, MetaConv2d, MetaLinear, MetaBatchNorm2d
from torchmeta.modules.utils import get_subdict
import torch.nn.functional as F


class MetaNetwork(MetaModule):
    def __init__(self):
        super().__init__()
        self.conv1 = MetaConv2d(in_channels=3, out_channels=1, kernel_size=3)
        self.fc1 = MetaLinear(in_features=30 * 22, out_features=3)

    def forward(self, x, params=None):
        batch_size = x.size()[0]
        out_conv1 = F.relu(self.conv1(x, params=get_subdict(params, "conv1")))
        flattened = out_conv1.view(batch_size, -1)
        out_fc1 = self.fc1(flattened, params=get_subdict(params, "fc1"))
        return out_fc1


class MetaAttentionNetworkCoord(MetaModule):
    def __init__(self):
        super().__init__()
        # spatial information is encoded as coord feature maps, one for x and one for y dimensions, fourth/fifth channels
        self.conv1 = MetaConv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.batch_norm1 = MetaBatchNorm2d(64, track_running_stats=False)
        self.conv2 = MetaConv2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1)
        self.batch_norm2 = MetaBatchNorm2d(32, track_running_stats=False)
        self.conv3 = MetaConv2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.batch_norm3 = MetaBatchNorm2d(16, track_running_stats=False)
        #self.fc1 = MetaLinear(in_features=384, out_features=64)
        self.fc1 = MetaLinear(in_features=32, out_features=64)
        self.fc2 = MetaLinear(in_features=64, out_features=64)
        self.fc3 = MetaLinear(in_features=64, out_features=6)

    def forward(self, image_batch, params=None):
        batch_size = image_batch.size()[0]
        out_conv1 = self.conv1.forward(image_batch, params=get_subdict(params, "conv1"))
        out_conv1 = F.relu(self.batch_norm1.forward(out_conv1, params=get_subdict(params, "batch_norm1")))
        out_conv2 = self.conv2.forward(out_conv1, params=get_subdict(params, "conv2"))
        out_conv2 = F.relu(self.batch_norm2.forward(out_conv2, params=get_subdict(params, "batch_norm2")))
        out_conv3 = self.conv3.forward(out_conv2, params=get_subdict(params, "conv3"))
        out_conv3 = F.relu(self.batch_norm3.forward(out_conv3, params=get_subdict(params, "batch_norm3")))
        out_conv3 = out_conv3.view(batch_size, -1)
        out_fc1 = F.relu(self.fc1.forward(out_conv3, params=get_subdict(params, "fc1")))
        out_fc2 = F.relu(self.fc2.forward(out_fc1, params=get_subdict(params, "fc2")))
        out_fc3 = self.fc3.forward(out_fc2, params=get_subdict(params, "fc3"))
        return out_fc3
