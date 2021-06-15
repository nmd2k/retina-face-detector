import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MobileNetV1, MobileNetV2, ResNet50

class RetinaFace(nn.Module):
    def __init__(self, backbone='MBNv1'):
        """
        Backbone: select backbone (default: MobileNetV1)
        """
        super(RetinaFace, self).__init__()

        # load backbone

        # class head + bbox head + landmark head
        
        pass

    def forward(self, input):
        pass