import torch
from torch import nn
import timm


def get_model(name_model, path_to_model, num_classes, pretrained=True):
    model = timm.create_model(name_model, pretrained=pretrained)
    model.head = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes))

    if path_to_model is not None:
        checkpoint = torch.load(path_to_model)
        model.load_state_dict(checkpoint)

    return model
