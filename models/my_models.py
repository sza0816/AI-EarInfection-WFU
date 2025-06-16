import torch.nn as nn
from torchvision import models as md
import timm

def get_model(name, num_classes = 4, weights = 'DEFAULT'):
    if name == "resnet34":
        w = "IMAGENET1K_V1" if weights == 'DEFAULT' else None
        model = md.resnet34(weights = w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnetb0":
        w = "IMAGENET1K_V1" if weights == 'DEFAULT' else None
        model = md.efficientnet_b0(weights = w)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "convnext":
        w = md.ConvNeXt_Tiny_Weights.DEFAULT if weights == 'DEFAULT' else None
        model = md.convnext_tiny(weights = w)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif name == "swint":
        w = md.Swin_T_Weights.DEFAULT if weights == 'DEFAULT' else None
        model = md.swin_t(weights = w)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif name == "vitbase16":
        w = md.ViT_B_16_Weights.DEFAULT if weights == 'DEFAULT' else None
        model = md.vit_b_16(weights = w)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif name == "efficientvitb0":
        # import timm
        model = timm.create_model('efficientvit_b0', pretrained = (weights == 'DEFAULT'))
        model.head.classifier[4] = nn.Linear(model.head.classifier[4].in_features, num_classes)
    else: 
        raise ValueError(f"Unsupported model: {name}")
    
    return model
