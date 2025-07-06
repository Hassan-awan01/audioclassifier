import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.use_shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1,stride=stride,padding=1,bias=False)
            )
    def forward()