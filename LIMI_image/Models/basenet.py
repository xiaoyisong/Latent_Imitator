"""ResNet

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

[2] https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):    
    def __init__(self, n_classes=1, pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, 512)
        self.relu = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )    

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        outputs = self.network(self.relu(features))

        return outputs, features

