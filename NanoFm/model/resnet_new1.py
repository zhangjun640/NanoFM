import torch
from torch import nn
import torch.nn.functional as F
from .util import FlattenLayer  # Assume util.py is in the same directory

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)


    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))


        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:

            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=1))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class GlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool1d, self).__init__()

    def forward(self, x):
        return F.avg_pool1d(x, kernel_size=x.size()[2:])


def build_resnet_backbone(in_channels=5, out_channels=256, initial_kernel_size=7, block_config=None):
    """
    Build a generic ResNet backbone.

    Parameters:
    - initial_kernel_size (int): Kernel size of the first convolution layer (e.g., 3 or 7).
    - block_config (list): A list of 4 integers defining the number of residual blocks
                           in the 4 ResNet stages.
                           (e.g., ResNet-18 is [2, 2, 2, 2], ResNet-34 is [3, 4, 6, 3])
    """
    if block_config is None:
        block_config = [2, 2, 2, 2]  # Default ResNet-18 structure

    if initial_kernel_size == 7:
        padding = 3
    elif initial_kernel_size == 3:
        padding = 1
    else:
        raise ValueError(f"Unsupported initial_kernel_size: {initial_kernel_size}")

    # Stem
    net = nn.Sequential(
        nn.Conv1d(in_channels, 32, kernel_size=initial_kernel_size, stride=2, padding=padding),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    )

    # ResNet stages
    net.add_module("resnet_block1", resnet_block(32, 32, block_config[0], first_block=True))
    net.add_module("dropout1", nn.Dropout(p=0.2))
    net.add_module("resnet_block2", resnet_block(32, 64, block_config[1]))
    net.add_module("resnet_block3", resnet_block(64, 128, block_config[2]))
    net.add_module("dropout2", nn.Dropout(p=0.2))
    net.add_module("resnet_block4", resnet_block(128, out_channels, block_config[3]))

    # Head
    net.add_module("global_avg_pool", GlobalAvgPool1d())
    return net


class ResNetModel(nn.Module):
    def __init__(self, model_config='resnet18_k7', in_channels=5, out_channels=256):
        """
        Unified ResNet model.

        Parameters:
        - model_config (str): A string defining the model architecture.
            - 'resnet18_k7': Original Resnet2 (Kernel=7, depth=18)
            - 'resnet18_k3': Original Resnet3 (Kernel=3, depth=18)
            - 'resnet34_k7': Deeper version of Resnet2 (Kernel=7, depth=34)
            - 'resnet34_k3': Deeper version of Resnet3 (Kernel=3, depth=34)
        """
        super(ResNetModel, self).__init__()

        # Define architecture configurations
        self.configs = {
            'resnet18_k7': {'kernel': 7, 'blocks': [2, 2, 2, 2]},
            'resnet18_k3': {'kernel': 3, 'blocks': [2, 2, 2, 2]},
            'resnet34_k7': {'kernel': 7, 'blocks': [3, 4, 6, 3]},
            'resnet34_k3': {'kernel': 3, 'blocks': [3, 4, 6, 3]},
        }

        if model_config not in self.configs:
            raise ValueError(f"Unknown model_config: {model_config}. Available options: {list(self.configs.keys())}")

        config = self.configs[model_config]

        # Build backbone network
        self.resnet = build_resnet_backbone(
            in_channels=in_channels,
            out_channels=out_channels,
            initial_kernel_size=config['kernel'],
            block_config=config['blocks']
        )

    def forward(self, x):
        x = self.resnet(x).view(x.shape[0], -1)
        return x


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.avg_pool1d(x, kernel_size=x.size()[2:])
        return x
