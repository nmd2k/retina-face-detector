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
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, leaky=0, activation=True):
        super(Conv_BN, self).__init__()

        self.conv  = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
        self.bn    = nn.BatchNorm2d(out_channels),
        self.leaky = nn.LeakyReLU(negative_slope=leaky, inplace=True)

        self.activation = activation

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        if self.activation:
            x = self.leaky(x)
        return x

class SSH(nn.Module):
    def __init__(self, in_channels, out_channels, leaky=0):
        super(SSH, self).__init__()
        assert out_channels%4 == 0
        self.conv1_1       = Conv_BN(in_channels, out_channels//2, activation=False)

        self.conv1_2      = Conv_BN(in_channels, out_channels//4)
        self.conv2_2      = Conv_BN(out_channels//4, out_channels//4, activation=False)

        self.conv2_3      = Conv_BN(out_channels//4, out_channels//4)
        self.conv3_3      = Conv_BN(out_channels//4, out_channels//4, activation=False)

    def forward(self, input):
        # block 1
        C1_1 = self.conv1_1(input)
        C1_2 = self.conv1_2(input)

        # block 2
        C2_2 = self.conv2_2(C1_2)
        C2_3 = self.conv2_3(C1_2)

        # block 3
        C3_3 = self.conv3_3(C2_3)

        out = torch.cat([C1_1, C2_2, C3_3], dim=1)
        return F.relu(out)

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

    def forward(self, input):
        pass

class MobileNetV1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1000, start_frame=32):
        super(MobileNetV1, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.start_frame  = start_frame

        self.stage1 = nn.Sequential(                            # Input channels-Output channels
            Conv_BN(in_channels, start_frame, stride=2, leaky=0.1),    # 3-32
            Conv_DW(start_frame, start_frame*2),                # 32-64
            Conv_DW(start_frame*2, start_frame*4, stride=2),    # 64-128
            Conv_DW(start_frame*4, start_frame*4),              # 128-128
            Conv_DW(start_frame*4, start_frame*8, stride=2),    # 128-256
            Conv_DW(start_frame*8, start_frame*8),              # 256-256
        )
        
        self.stage2 = nn.Sequential(
            Conv_DW(start_frame*8, start_frame*16, stride=2),   # 256-512
            Conv_DW(start_frame*16, start_frame*16),            # 512-512
            Conv_DW(start_frame*16, start_frame*16),            # 512-512
            Conv_DW(start_frame*16, start_frame*16),            # 512-512
            Conv_DW(start_frame*16, start_frame*16),            # 512-512
            Conv_DW(start_frame*16, start_frame*16),            # 512-512
        )

        self.stage3 = nn.Sequential(
            Conv_DW(start_frame*16, start_frame*32, stride=2),  # 512-1024
            Conv_DW(start_frame*32, start_frame*32)             # 1024-1024
        )

        self.avg    = nn.AdaptiveAvgPool2d((1,1))
        self.fc     = nn.Linear(start_frame*32, out_channels)   # 1024-1000

    def forward(self, input):
        x = self.stage1(input)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)

        x = x.view(-1, self.start_frame*32)
        x = self.fc(x)
        
        return x

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
