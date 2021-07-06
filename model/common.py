import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_DW(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=0.1):
        super(Conv_DW, self).__init__()
        self.conv1  = nn.Conv2d(in_channels, in_channels, 3, stride, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(in_channels)

        self.conv2  = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2    = nn.BatchNorm2d(out_channels)
        
        self.leaky  = nn.LeakyReLU(negative_slope=leaky, inplace=True)

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

        self.conv  = nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False)
        self.bn    = nn.BatchNorm2d(out_channels)
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
    def __init__(self, in_channels_list, out_channels, leaky=0):
        """
        Module that adds a FPN from on top of a set of feature maps. This is based on
        `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

        The feature maps are currently supposed to be in increasing depth
        order.

        The input to the model is expected to be an OrderedDict[Tensor], containing
        the feature maps on top of which the FPN will be added.

        Args:
            in_channels_list (list[int]): number of channels for each feature map that
                is passed to the module
            out_channels (int): number of channels of the FPN representation
        """
        super(FPN, self).__init__()
        assert len(in_channels_list) == 4
        # [IN_CHANNELS*2, IN_CHANNELS*4, IN_CHANNELS*8, IN_CHANNELS*16]
        self.layer_feature_1 = Conv_BN(in_channels_list[0], out_channels, 1, padding=0, leaky=leaky)
        self.layer_feature_2 = Conv_BN(in_channels_list[1], out_channels, 1, padding=0, leaky=leaky)
        self.layer_feature_3 = Conv_BN(in_channels_list[2], out_channels, 1, padding=0, leaky=leaky)
        self.layer_feature_4 = Conv_BN(in_channels_list[3], out_channels, 1, padding=0, leaky=leaky)
        self.layer_feature_5 = Conv_BN(in_channels_list[3], out_channels, 3, stride=2, leaky=leaky)

        self.merge           = Conv_BN(out_channels, out_channels, leaky=leaky)
        self.upsample        = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists
        names   = list(input.keys())
        input   = list(input.values())

        output1 = self.layer_feature_1(input[0])
        output2 = self.layer_feature_2(input[1])
        output3 = self.layer_feature_3(input[2])
        output4 = self.layer_feature_4(input[3])
        output5 = self.layer_feature_5(input[3])

        # output5 = self.upsample(output5)
        # output4 = self.upsample(output4)

        up4     = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="nearest")
        output3 = output3 + up4
        # output3 = self.upsample(output3)
        output3 = self.merge(output3)

        up3     = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        # output2 = self.upsample(output2)
        output2 = self.merge(output2)

        up2     = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        # output1 = self.upsample(output1)
        output1 = self.merge(output1)

        return [output1, output2, output3, output4, output5]

class MobileNetV1(nn.Module):
    def __init__(self, in_channels=3, out_channels=1000, start_frame=32):
        super(MobileNetV1, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.start_frame  = start_frame

        self.layer2 = nn.Sequential(                            # Input channels-Output channels
            Conv_BN(in_channels, start_frame, stride=2, leaky=0.1),    # 3-32
            Conv_DW(start_frame, start_frame*2),                # 32-64
            Conv_DW(start_frame*2, start_frame*4, stride=2),    # 64-128
            Conv_DW(start_frame*4, start_frame*4),              # 128-128
            Conv_DW(start_frame*4, start_frame*8, stride=2),    # 128-256
            Conv_DW(start_frame*8, start_frame*8),              # 256-256
        )
        
        self.layer3 = nn.Sequential(
            Conv_DW(start_frame*8, start_frame*16, stride=2),   # 256-512
            Conv_DW(start_frame*16, start_frame*16),            # 512-512
            Conv_DW(start_frame*16, start_frame*16),            # 512-512
            Conv_DW(start_frame*16, start_frame*16),            # 512-512
            Conv_DW(start_frame*16, start_frame*16),            # 512-512
            Conv_DW(start_frame*16, start_frame*16),            # 512-512
        )

        self.layer4 = nn.Sequential(
            Conv_DW(start_frame*16, start_frame*32, stride=2),  # 512-1024
            Conv_DW(start_frame*32, start_frame*32)             # 1024-1024
        )

        self.avg    = nn.AdaptiveAvgPool2d((1,1))
        self.fc     = nn.Linear(start_frame*32, out_channels)   # 1024-1000

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg(x)

        x = x.view(-1, self.start_frame*32)
        x = self.fc(x)
        
        return x