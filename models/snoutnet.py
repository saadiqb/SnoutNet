
# models/snoutnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class SnoutNet(nn.Module):
    """
    Conv(3->64, 3x3, padding=1) -> ReLU -> MaxPool(k=3,s=4,ceil)
    Conv(64->128, 3x3, padding=1) -> ReLU -> MaxPool(k=3,s=4,ceil)
    Conv(128->256, 3x3, padding=1) -> ReLU -> MaxPool(k=3,s=4,ceil)
    Flatten (4x4x256=4096) -> FC 4096->1024 -> ReLU
    FC 1024->1024 -> ReLU
    FC 1024->2  (regresses (u, v) in pixels for 227x227 inputs)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=4, ceil_mode=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=4, ceil_mode=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=4, ceil_mode=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    summary(SnoutNet(), input_size=(1, 3, 227, 227), col_names=("output_size", "num_params"))
