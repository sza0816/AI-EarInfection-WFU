from torchvision import models as torchvision_models
from .my_models import get_convnext

models = torchvision_models

models.get_convnext = get_convnext