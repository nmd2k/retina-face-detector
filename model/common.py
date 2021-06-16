import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_DW(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=0.1):
        super(Conv_DW, self).__init__()
        self.conv1  = nn.Conv2d(in_channels, in_channels, 3, stride, 1, bias=False),
        self.bn1    = nn.BatchNorm2d(in_channels),

        self.conv2  = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        self.bn2    = nn.BatchNorm2d(out_channels),
        
        self.leaky  = nn.LeakyReLU(negative_slope=leaky, inplace=True),

    def forward(self, input):
        # Depthwise conv block
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.leaky(x)

        # Pointwise conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky(x)

        return x

class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=0, activation=True):
        super(Conv_BN, self).__init__()

        self.conv  = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        self.bn    = nn.BatchNorm2d(out_channels),
        self.leaky = nn.LeakyReLU(negative_slope=leaky, inplace=True)

        self.activation = activation

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        if self.activation:
            x = self.leaky(x)
        return x

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        pass

    def forward(self, input):
        pass

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        pass

    def forward(self, input):
        pass

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        pass

    def forward(self, input):
        pass
