import torch
from torch import nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, simple=False, bilinear=False):
        """
        The U-Net architecture.
        
        :param n_channels: Number of input channels (e.g., 3 for RGB images)
        :param n_classes: Number of output classes (e.g., 1 for binary segmentation)
        :param simple: If True, creates a smaller U-Net with fewer layers.
        :param bilinear: If True, use bilinear upsampling instead of transposed convolutions.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.simple = simple
        
        factor = 2 if bilinear else 1

        if not self.simple:
            self.inc = (DoubleConv(n_channels, 64))
            self.down1 = (Down(64, 128))
            self.down2 = (Down(128, 256))
            self.down3 = (Down(256, 512))
            self.down4 = (Down(512, 1024 // factor))
            self.up1 = (Up(1024, 512 // factor, bilinear))
            self.up2 = (Up(512, 256 // factor, bilinear))
            self.up3 = (Up(256, 128 // factor, bilinear))
            self.up4 = (Up(128, 64, bilinear))
            self.outc = (OutConv(64, n_classes))
        else:
            self.inc = (DoubleConv(n_channels, 64))
            self.down1 = (Down(64, 128))
            self.down2 = (Down(128, 256 // factor))
            self.up1 = (Up(256, 128 // factor, bilinear))
            self.up2 = (Up(128, 64, bilinear))
            self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        if not self.simple:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up1(x3, x2)
            x = self.up2(x, x1)
            logits = self.outc(x)
            
        return logits

    def use_checkpointing(self):
        """
        Enable gradient checkpointing to save memory, but at the cost of additional computation during backpropagation.
        """
        if not self.simple:
            self.inc = torch.utils.checkpoint.checkpoint(self.inc)
            self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
            self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
            self.down3 = torch.utils.checkpoint.checkpoint(self.down3)
            self.down4 = torch.utils.checkpoint.checkpoint(self.down4)
            self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
            self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
            self.up3 = torch.utils.checkpoint.checkpoint(self.up3)
            self.up4 = torch.utils.checkpoint.checkpoint(self.up4)
            self.outc = torch.utils.checkpoint.checkpoint(self.outc)
        else:
            self.inc = torch.utils.checkpoint.checkpoint(self.inc)
            self.down1 = torch.utils.checkpoint.checkpoint(self.down1)
            self.down2 = torch.utils.checkpoint.checkpoint(self.down2)
            self.up1 = torch.utils.checkpoint.checkpoint(self.up1)
            self.up2 = torch.utils.checkpoint.checkpoint(self.up2)
            self.outc = torch.utils.checkpoint.checkpoint(self.outc)