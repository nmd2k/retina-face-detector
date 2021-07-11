import torch
import numpy as np
import torch.nn as nn

from model.config import *
from utils.box_utils import point_form

class Anchors(nn.Module):
    def __init__(self, image_size=None, feat_shape=None, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        # init param
        self.pyramid_levels = pyramid_levels
        self.strides        = strides
        self.ratios         = ratios
        self.scales         = scales
        self.sizes          = sizes
        self.feat_shape     = feat_shape
        self.image_size     = image_size

        if pyramid_levels is None:
            # the original pyramid net have P2, P3, P4, P5, P6, C7/P7
            self.pyramid_levels = [3, 4, 5, 6, 7]
        
        if strides is None:
            self.strides = [2 ** (x) for x in self.pyramid_levels]

        if sizes is None:
            self.sizes = [2 ** (x+1) for x in self.pyramid_levels]

        if ratios is None:
            # most of bounding box in wider face were 1/2 and 2 in ratio aspect
            self.ratios = [0.5, 1]

        if scales is None:
            # defaul scale defined in paper is 2^(1/3)
            self.scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        if image_size is None: 
            self.image_size = np.array([INPUT_SIZE, INPUT_SIZE])

        if feat_shape is None:
            self.feat_shape = [(self.image_size + x - 1) // x for x in self.strides]
        
    def forward(self):
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            anchors         = torch.from_numpy(anchors).to(dtype=torch.float)
            shifted_anchors = shift(self.feat_shape[idx], self.strides[idx], anchors)
            # from IPython import embed
            # embed()
            # shifted_anchors = shifted_anchors/self.image_size[0]
            shifted_anchors[:, 0::2] = shifted_anchors[:, 0::2]/self.image_size[0]
            shifted_anchors[:, 1::2] = shifted_anchors[:, 1::2]/self.image_size[1]

            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = torch.from_numpy(all_anchors).to(dtype=torch.float)

        return all_anchors

def generate_anchors(num_anchors=None, base_size=8, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = [0.5, 1, 2]

    if scales is None:
        scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    if num_anchors == None:
        num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 3] = np.sqrt(areas / np.repeat(ratios, len(scales))) # h
    anchors[:, 2] = anchors[:, 3] * np.repeat(ratios, len(scales))  # w

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    # anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    # anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    # keep it form (0, 0, w, h)
    return anchors

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]

    re_anchors  = anchors.reshape((1, A, 4))
    shifted     = shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    shifted[:,:,2:] = 0 # format (x_c, y_c, w, h) need to maintain w, h

    all_anchors = re_anchors + shifted
    all_anchors = all_anchors.reshape((K * A, 4))
    all_anchors = point_form(all_anchors)

    return all_anchors