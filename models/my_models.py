import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny, convnext_small, convnext_base, convnext_large, 
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
)


# for flexibly choosing convnext model
def get_convnext(size = "tiny", weights = "DEFAULT", num_classes = 4):
    model_map={
        "tiny": (convnext_tiny, ConvNeXt_Tiny_Weights),
        "small":(convnext_small, ConvNeXt_Small_Weights),
        "base":(convnext_base,ConvNeXt_Base_Weights),
        "large":(convnext_large,ConvNeXt_Large_Weights)
    }

    # raise error if model not found
    if size not in model_map: 
        raise ValueError(f"Invalid model size: {size}. Choose from 'tiny', 'small', 'base', 'large'. ")

    model_fn, weight_enum = model_map[size]         # select model from map

    if weights == "DEFAULT":                         # check weights
        weights = weight_enum.DEFAULT
    elif weights is None:
        weights = None
    else:
        raise ValueError("weights must be 'DEFAULT' or None.")

    model = model_fn(weights = weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)        ##
    return model

