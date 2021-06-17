import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MobileNetV1, MobileNetV2, ResNet50

class ClassHead(nn.Module):
    def __init__(self):
        """
        Face classification 
        """
        super().__init__()

    def forward(self, input):

        pass

class BboxHead(nn.Module):
    def __init__(self):
        """
        Face bounding box
        """
        super().__init__()

    def forward(self, input):
        pass

class LandmarkHead(nn.Module):
    def __init__(self):
        """
        Facial landmark
        """
        super().__init__()

    def forward(self, input):
        pass

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