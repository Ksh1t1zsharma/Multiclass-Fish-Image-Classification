import torch.nn as nn
import torchvision.models as models

def get_model(name="resnet18", num_classes=2, pretrained=True):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(pretrained=pretrained)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(pretrained=pretrained)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    elif name == "mobilenet_v2":
        m = models.mobilenet_v2(pretrained=pretrained)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Model {name} not implemented. Choose resnet18/resnet50/mobilenet_v2.")
    return m
