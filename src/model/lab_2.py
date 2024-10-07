import torch

from torch import nn

from lib.model.classifier_module import ClassifierModule


class Lab2Teacher(ClassifierModule):

    def __init__(self, device, num_classes=1000, dropout=0.5):
        super(Lab2Teacher, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 2024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(2024, num_classes),
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.classifier(x)


class Lab2Student(ClassifierModule):

    def __init__(self, device, num_classes=1000, dropout=0.5):
        super(Lab2Student, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, num_classes),
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.classifier(x)
