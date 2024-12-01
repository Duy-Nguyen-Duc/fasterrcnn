"""
Faster RCNN model with the VGG16 backbone from Torchvision.
Torchvision link: https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16_bn.html#torchvision.models.VGG16_BN_Weights
ResNet paper: https://arxiv.org/abs/1409.1556
"""

import torchvision
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN

from torchvision.models import vgg16
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork as FPN
from torchvision.ops import misc as misc_nn_ops

def vgg_fpn_backbone(pretrained=True):
    """
    Builds a VGG16 backbone with an FPN (Feature Pyramid Network).
    """
    # Load pretrained VGG16 from torchvision
    vgg = vgg16(pretrained=pretrained)
    features = list(vgg.features.children())

    # Divide VGG16 into stages
    stages = [
        nn.Sequential(*features[:5]),  # Stage 1: up to ReLU_2 (conv1_2)
        nn.Sequential(*features[5:10]),  # Stage 2: up to ReLU_4 (conv2_2)
        nn.Sequential(*features[10:17]),  # Stage 3: up to ReLU_7 (conv3_3)
        nn.Sequential(*features[17:24]),  # Stage 4: up to ReLU_10 (conv4_3)
        nn.Sequential(*features[24:]),  # Stage 5: up to ReLU_13 (conv5_3)
    ]

    # Freeze early layers if required
    for name, parameter in vgg.named_parameters():
        if 'features.0' in name or 'features.5' in name:  # Example: freeze initial layers
            parameter.requires_grad_(False)

    # Define return layers and channel configurations
    return_layers = {
        'vgg4': 4,
    }

    # Input channels for each stage
    in_channels_list = [512]
    out_channels = 256  # Output channels for the FPN

    # Register stages as named submodules for compatibility with FPN
    backbone = nn.Module()
    for i, stage in enumerate(stages):
        stage_name = f"vgg{i}"
        setattr(backbone, stage_name, stage)

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)



def create_model(num_classes, pretrained=True, coco_model=False):
    # Load the pretrained ResNet18 backbone.
    backbone = vgg_fpn_backbone(pretrained)
    
    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 512 for ResNet18.

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
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)