import torch
import torch.nn as nn
from scipy import io
import numpy as np
import torch.nn.functional as F
from loss import *
from metric import *

class Unet(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.cnn1 = double_conv(channels, 64, 10)
        self.down1 = down_double(64, 128, 8)
        self.down2 = down_triple(128, 256, 6)
        self.down3 = down_triple(256, 512, 4)
        self.down4 = down_triple(512, 512, 2)
        self.up1 = up_triple(1024, 256, 4)
        self.up2 = up_triple(512, 128, 6)
        self.up3 = up_triple_not(256, 64, 8)
        self.up4 = up_double(128, 64, 10)
        self.outc = outconv(64, classes)
        self.dropout = nn.Dropout2d(0.2,inplace=False)
        # self.sm = nn.Sigmoid()

    def forward(self, x):
        # x =
        x1 = self.cnn1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        outcx = self.outc(x)
        outcx = self.dropout(outcx)
        # y = self.sm(x)
        # result = torch.argmax(x, dim = 1, keepdim = True)
        return outcx

    def criterion(self, logit, truth):
        loss = bce_dice_loss(F.sigmoid(logit), truth)
        return loss

    def metric(self, logit, truth, multi_class=False, threshold=0.5):

        prob = F.sigmoid(logit)

        dice = dice_accuracy(prob, truth, threshold=threshold, is_average=True)
        Jac = Jaccard_accuracy(prob, truth, threshold=threshold, is_average=True)

        return dice, Jac

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, i):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            SELayer(out_ch, i),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            SELayer(out_ch, i)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class triple_conv(nn.Module):
    '''(conv => BN => ReLU) * 3'''

    def __init__(self, in_ch, out_ch, i):
        super(triple_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            SELayer(out_ch, i),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            SELayer(out_ch, i),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            SELayer(out_ch, i)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down_double(nn.Module):
    def __init__(self, in_ch, out_ch, i):
        super(down_double, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, i)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class down_triple(nn.Module):
    def __init__(self, in_ch, out_ch, i):
        super(down_triple, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            triple_conv(in_ch, out_ch, i)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_double(nn.Module):
    def __init__(self, in_ch, out_ch, i):
        super(up_double, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 4, stride=2)
        self.conv = double_conv(in_ch // 2, out_ch, i)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = self.conv(x1)
        return x


class up_triple(nn.Module):
    def __init__(self, in_ch, out_ch, i):
        super(up_triple, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 4, stride=2)
        self.conv = triple_conv(in_ch, out_ch, i)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class up_triple_not(nn.Module):
    def __init__(self, in_ch, out_ch, i):
        super(up_triple_not, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 4, stride=2)
        self.conv = triple_conv(in_ch // 2, out_ch, i)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = self.conv(x1)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, i, reduction=16):
        super(SELayer, self).__init__()
        self.num = i
        self.avg_pool = nn.AdaptiveAvgPool2d(self.num)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels // reduction, out_channels=channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c, self.num, self.num)
        y = self.fc(y).view(b, c, self.num, self.num)
        y = nn.functional.upsample_bilinear(y, [h, w])
        return x * y
