import torch
import torchvision
from lib.model.classifier_module import ClassifierModule
from torch import nn


class Inception(ClassifierModule):

    def __init__(self, device, num_classes=1000, dropout=0.5):
        super(Inception, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.net = torchvision.models.inception_v3(
            num_classes=num_classes, dropout=dropout
        )
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.net(x)[0]
        else:
            return x


class CoTraining(ClassifierModule):

    def __init__(self, device, modules: list[nn.Module], num_classes=1000):
        super(CoTraining, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.co_modules = nn.ModuleList(modules)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.stack([it(x) for it in self.co_modules])

        return torch.sum(y, dim=0)
