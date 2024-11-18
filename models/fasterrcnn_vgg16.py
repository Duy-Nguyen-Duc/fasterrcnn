"""
Faster RCNN model with the VGG16 backbone from Torchvision.
Torchvision link: https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16_bn.html#torchvision.models.VGG16_BN_Weights
ResNet paper: https://arxiv.org/abs/1409.1556
"""

import torchvision
import torch.nn as nn

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from .gcn_lib import Grapher, act_layer
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x
def create_model(num_classes, pretrained=True, coco_model=False):
    # Load the pretrained ResNet18 backbone.
    model_backbone = torchvision.models.vgg16_bn(weights='DEFAULT')

    backbone = model_backbone.features
    layers = []
    for _ in range(1):
        layers.append(
            Grapher(
                in_channels=64, kernel_size=9, dilation=1, conv="mr", act="gelu",
                norm="batch", bias=True, stochastic=False, epsilon=0.2,
                r=1, n=196, relative_pos=False
            )
        )
        layers.append(FFN(64, 64 * 4, act="gelu"))
    
    new_backbone = nn.Sequential(
        *model_backbone.features[0:7],
        nn.Sequential(*layers),
        *model_backbone.features[7:],
    )
    # We need the output channels of the last convolutional layers from
    # the features for the Faster RCNN model.
    # It is 512 for ResNet18.
    new_backbone.out_channels = 512

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
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=new_backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)