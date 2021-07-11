import torch
import torch.nn as nn
from math import sqrt, pow
import torch.nn.functional as F

from model.config import *
from model._utils import IntermediateLayerGetter
from model.common import FPN, SSH, MobileNetV1

class ClassHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        """
        Face classification 
        """
        super(ClassHead, self).__init__()
        self.conv        = nn.Conv2d(in_channels, num_anchors*2, 1)

    def forward(self, input):
        out = self.conv(input)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        """
        Face bounding box
        """
        super(BboxHead, self).__init__()
        self.conv        = nn.Conv2d(in_channels, num_anchors*4, 1)

    def forward(self, input):
        out = self.conv(input)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self, in_channels, num_anchors=3):
        """
        Facial landmark
        """
        super(LandmarkHead, self).__init__()
        # 5 (x, y) refer to coordinate of 5 landmarks
        self.conv = nn.Conv2d(in_channels, num_anchors*10, 1)

    def forward(self, input):
        out = self.conv(input)
        out = out.permute(0,2,3,1).contiguous()
        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, model_name='resnet50', freeze_backbone=False, pretrain_path=None, is_train=True):
        """
        Model RetinaFace for face recognition based on:
        `"RetinaFace: Single-stage Dense Face Localisation in the Wild" <https://arxiv.org/abs/1905.00641>`_.
        """
        super(RetinaFace, self).__init__()
        self.is_train = is_train
        # load backbone
        backbone = None
        if model_name == 'mobilenet0.25':
            backbone            = MobileNetV1(start_frame=START_FRAME)
            return_feature      = RETURN_MAP_MOBN1
            self.feature_map    = FEATURE_MAP_MOBN1
            
            if not pretrain_path is None:
                pretrain_weight = torch.load(pretrain_path)
                backbone.load_state_dict(pretrain_weight)

        elif model_name == 'mobilenetv2':
            return_feature = FEATURE_MAP_MOBN2
            backbone = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        
        elif 'resnet' in model_name:
            import torchvision.models as models
            if '18' in  model_name:
                backbone = models.resnet18(pretrained=True)

            elif '34' in  model_name:
                backbone = models.resnet34(pretrained=True)

            elif '50' in model_name:
                backbone = models.resnet50(pretrained=True)
            
            return_feature      = RETURN_MAP
            self.feature_map    = FEATURE_MAP

        else:
            print(f'Unable to select {model_name}.')

        num_fpn             = len(self.feature_map)

        # frozen pre-trained backbone
        self.body = IntermediateLayerGetter(backbone, return_feature)

        if freeze_backbone:
            for param in self.body.parameters():
                param.requires_grad = False
            print('\tBackbone freezed')

        in_channels_list = [IN_CHANNELS*2, IN_CHANNELS*4, IN_CHANNELS*8, IN_CHANNELS*16]
        self.fpn = FPN(in_channels_list=in_channels_list, out_channels=OUT_CHANNELS)
        self.ssh = SSH(in_channels=OUT_CHANNELS, out_channels=OUT_CHANNELS)

        # class head + bbox head + landmark head
        self.ClassHead      = self._make_class_head(inchannels=OUT_CHANNELS, anchor_num=6, fpn_num=num_fpn)
        self.BboxHead       = self._make_bbox_head(inchannels=OUT_CHANNELS, anchor_num=6, fpn_num=num_fpn)
        self.LandmarkHead   = self._make_landmark_head(inchannels=OUT_CHANNELS, anchor_num=6, fpn_num=num_fpn)

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, input):
        """
        The input to the RetinaFace is expected to be a Tensor

        Args:
            input (Tensor): image(s) for feed forward
        """
        out = self.body(input)

        # Feature Pyramid Net
        fpn = self.fpn(out)

        # Single-stage headless
        feature_1 = self.ssh(fpn[0])
        feature_2 = self.ssh(fpn[1])
        feature_3 = self.ssh(fpn[2])
        feature_4 = self.ssh(fpn[3])
        feature_5 = self.ssh(fpn[4])
        features = [feature_1, feature_2, feature_3, feature_4, feature_5]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications  = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions  = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.is_train:
            output = (bbox_regressions, classifications, ldm_regressions)
        else: 
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)

        return output

def forward(model, input, targets, anchors, loss_function, optimizer):
    """Due to the probability of OOM problem can happen, which might
    cause the "CUDA out of memory". I've passed all require grad into
    a function to free it while there is nothing refer to it.
    """
    predict = model(input)
    loss_l, loss_c, loss_landm = loss_function(predict, anchors, targets)
    loss = 1.3*loss_l + loss_c + loss_landm

    loss_l      = loss_l.item()
    loss_c      = loss_c.item()
    loss_landm  = loss_landm.item()

    # zero the gradient + backprpagation + step
    optimizer.zero_grad()

    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

    loss.backward()
    optimizer.step()

    del predict
    del loss

    return loss_l, loss_c, loss_landm
