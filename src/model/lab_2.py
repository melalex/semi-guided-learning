import torch

from torch import nn

from lib.model.classifier_module import ClassifierModule
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class Lab2Teacher(ClassifierModule):

    def __init__(self, device, num_classes=1000, dropout=0.5):
        super(Lab2Teacher, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.net = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT, dropout=dropout).to(
            device
        )
        self.net.avgpool = nn.Identity().to(device)

        for p in self.net.parameters():
            p.requires_grad = False

        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Lab2Student(ClassifierModule):

    def __init__(self, device, num_classes=1000, dropout=0.5):
        super(Lab2Student, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.net = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT, dropout=dropout).to(
            device
        )
        self.net.avgpool = nn.Identity().to(device)

        for p in self.net.parameters():
            p.requires_grad = False

        self.net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
