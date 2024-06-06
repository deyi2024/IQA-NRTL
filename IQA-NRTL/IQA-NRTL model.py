import torch
import torch.nn as nn
import torchvision.models as models

class IQA_NRTL(nn.Module):
    def __init__(self, num_classes=1):
        super(IQA_NRTL, self).__init__()
        # Use a pre-trained ResNet-50 model
        self.base_model = models.resnet50(pretrained=True)
        # Replace the final fully connected layer
        self.base_model.fc = nn.Identity()

        # Multi-scale information extraction
        self.multi_scale1 = nn.Conv2d(2048, 512, kernel_size=1)
        self.multi_scale2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.multi_scale3 = nn.Conv2d(128, 32, kernel_size=3, padding=1)

        # Perception module
        self.perception_module = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Adaptive fusion module
        self.adaptive_fusion = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.multi_scale1(x)
        x = self.multi_scale2(x)
        x = self.multi_scale3(x)
        x = self.perception_module(x)
        x = self.adaptive_fusion(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
