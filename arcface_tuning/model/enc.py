import torch
import torch.nn as nn

class FreezeEncoder(nn.Module):
    def __init__(self, backbone):
        self.backbone = backbone

    def forward(self, x, freeze = True):
        if freeze:
            with torch.no_grad():
                return self.backbone(x)
        else:
            return self.backbone(x)

    def get_backbone(self):
        return self.backbone