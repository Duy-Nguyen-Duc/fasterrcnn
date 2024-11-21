"""
A lot of scripts borrowed/adapted from Detectron2.
https://github.com/facebookresearch/detectron2/blob/38af375052d3ae7331141bc1a22cfa2713b02987/detectron2/modeling/backbone/backbone.py#L11
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torchvision

from functools import partial
from torchvision.models.detection import FasterRCNN
from models.layers import (
    Backbone, 
    PatchEmbed, 
    Block, 
    get_abs_pos,
    get_norm,
    Conv2d,
    LastLevelMaxPool
)
from models.utils import _assert_strides_are_log2_contiguous

class ViG(Backbone):
    def __init__(self):
        super().__init__()

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4