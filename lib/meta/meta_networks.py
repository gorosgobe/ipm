from torch import nn
from torchmeta.modules import MetaModule, MetaSequential, MetaConv2d, MetaLinear
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
