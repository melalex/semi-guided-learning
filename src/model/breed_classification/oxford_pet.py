from torch import nn
import torch
from torchvision.models import vgg16, VGG16_Weights

from lib.model.classifier_module import ClassifierModule


class OxfordPet(ClassifierModule):

    def __init__(self, device):
        super(OxfordPet, self).__init__()
        self.device = device
        self.num_classes = 37
        
        self.net = vgg16(weights=VGG16_Weights.DEFAULT).to(device)
        self.out_layer = nn.Linear(1000, self.num_classes)

        for p in self.net.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.net(x)
        return self.out_layer(x1)
