"""
Faster RCNN model with the VGG16 backbone from Torchvision.
Torchvision link: https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16_bn.html#torchvision.models.VGG16_BN_Weights
ResNet paper: https://arxiv.org/abs/1409.1556
"""

import torchvision
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models import vgg16
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork as FPN, LastLevelMaxPool

class VGG16FPNBackbone(nn.Module):
    """
    A VGG16 backbone with FPN for feature extraction using a pretrained VGG16 from torchvision.
    """
    def __init__(self, pretrained=True, in_channels_list=None, return_layers=None, out_channels=256):
        super().__init__()

        # Load pretrained VGG16 from torchvision
        vgg = vgg16(pretrained=pretrained)
        
        # Extract feature layers from the pretrained model
        features = list(vgg.features.children())
        self.stages = [
            nn.Sequential(*features[:5]),  # Stage 1: up to ReLU_2 (conv1_2)
            nn.Sequential(*features[5:10]),  # Stage 2: up to ReLU_4 (conv2_2)
            nn.Sequential(*features[10:17]),  # Stage 3: up to ReLU_7 (conv3_3)
            nn.Sequential(*features[17:24]),  # Stage 4: up to ReLU_10 (conv4_3)
            nn.Sequential(*features[24:]),  # Stage 5: up to ReLU_13 (conv5_3)
        ]

        # Add stage names for reference
        self._stage_names = [f"vgg{i}" for i in range(len(self.stages))]
        
        if in_channels_list is None:
            in_channels_list = [64, 128, 256, 512, 512]  # Default for VGG16

        # Define FPN
        self.fpn = FPN(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )

        if return_layers is None:
            self.return_layers = {f"vgg{i}": str(i) for i in range(len(self.stages))}
        else:
            self.return_layers = return_layers

        # Register stages as named modules
        for name, stage in zip(self._stage_names, self.stages):
            self.add_module(name, stage)

    def forward(self, x):
        # Forward pass through VGG stages
        features = {}
        for name, stage in zip(self._stage_names, self.stages):
            x = stage(x)
            if name in self.return_layers:
                features[self.return_layers[name]] = x

        # Pass features through FPN
        fpn_features = self.fpn(features)
        return fpn_features



def create_model(num_classes, pretrained=True, coco_model=False):
    # Load the pretrained ResNet18 backbone.
    return_layers = {'vgg0': '0', 'vgg1': '1', 'vgg2': '2', 'vgg3': '3', 'vgg4': '4'}
    out_channels = 256
    vgg16_fpn_backbone = VGG16FPNBackbone(
    pretrained=True,  # Load pretrained weights
    in_channels_list=None,
    return_layers=return_layers,
    out_channels=out_channels
    )
    
    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 512 for ResNet18.
    vgg16_fpn_backbone.out_channels = 512

    # Generate anchors using the RPN. Here, we are using 5x3 anchors.
    # Meaning, anchors with 5 different sizes and 3 different aspect 
    # ratios.
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2'],
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=vgg16_fpn_backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)