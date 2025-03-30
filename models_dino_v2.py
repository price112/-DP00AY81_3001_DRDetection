import torch
import torch.nn as nn
import timm
from timm.models import register_model


@register_model
def dinov2_vanliia(num_classes=5, drop_path_rate=0.2, pretrained=True, **kwargs):

    DinoV2CLS = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=pretrained,
        img_size=224,
        num_classes = num_classes,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    return DinoV2CLS

if __name__ == '__main__':
    Model = dinov2_vanliia()
    tensor = torch.randn(1, 3, 224, 224)
    Model.eval()
    print(Model(tensor))
