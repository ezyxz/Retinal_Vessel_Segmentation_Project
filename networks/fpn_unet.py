'''FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.unet_parts import *

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1): #64,512,1，#1024,512,1 #1024,512,2
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        temp_2 = self.conv2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        temp = self.shortcut(x)
        out += self.shortcut(x)

        out = F.relu(out)

        return out


class FPN(nn.Module):
    def __init__(self, input_bands, block, num_blocks): # 3 Bottleneck [2,2,2,2]
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(input_bands, 64, kernel_size=3, stride=1, padding=1, bias=False) #3
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 512, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 512, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1) #[1,1] [2,1] [2,1] [2,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride)) #64,512,1   1024
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''
        Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c0 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c0, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)

        c3 = self.layer2(c2)

        c4 = self.layer3(c3)

        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return c0, p2, p3, p4, p5

class fpn_unet(nn.Module):
    def __init__(self, input_bands, n_classes, bilinear=True): # 3 2
        super(fpn_unet, self).__init__()

        self.fpn = FPN(input_bands, Bottleneck, [2,2,2,2])
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        x1, x2, x3, x4, x5 = self.fpn(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# x = torch.randn((2, 3, 256, 256))
# model = fpn_unet(3, 5)
# y = model(x)
# print(y.shape)