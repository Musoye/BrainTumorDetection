import torch.nn as nn
from torchvision import models


def get_resnet_model(num_classes=2, pretrained=True):
    
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features

    model.fc = nn.Linear(num_features, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    return model
