import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
from PIL import Image
from skimage.measure import compare_psnr as psnr
import scipy.io as sio
import cv2
import os
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=5):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.0924, 0.1192, 0.0924],
                  [0.1192, 0.1536, 0.1192],
                  [0.0924, 0.1192, 0.0924]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, stride=1, padding=1, groups=self.channels)
        return x


class HighpassConv(nn.Module):
    def __init__(self, channels=5):
        super(HighpassConv, self).__init__()
        self.channels = channels
        kernel = [[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, stride=1, padding=1, groups=self.channels)
        return x


class B(nn.Module):
    def __init__(self, in_channel, filter):
        super(B, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, filter, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter, filter, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        x = self.relu(x)
        return x


class basiclayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(basiclayer, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=4*growth_rate, kernel_size=1, stride=1, padding=0,
                      dilation=1, bias=False),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=4 * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1,
                  dilation=1, bias=False),
            nn.ReLU(),
        )
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer3(out)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.ReLU(),
        )
    def forward(self, x):
        out = self.conv(x)
        out = F.avg_pool2d(out, 2)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, Layers):
        super(DenseBlock, self).__init__()
        layers = []
        channel = in_channels
        for i in range(Layers):
            layers.append(basiclayer(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x


class SE(nn.Module):
    def __init__(self, ch, re=2):
        super(SE, self).__init__()
        self.squeeze = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.excitation = nn.Sequential(
            nn.Conv2d(ch, ch//re, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch, ch//re, kernel_size=1, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        se = self.squeeze(x)
        se = self.excitation(se)
        x = x*se
        return x


class SCSE(nn.Module):
    def __init__(self, ch, re=2):
        super(SCSE, self).__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re, kernel_size=1, bias=False),
                                 nn.ReLU(),
                                 nn.Conv2d(ch//re, ch, kernel_size=1, bias=False),
                                 nn.Sigmoid())
        self.sSE = nn.Sequential(nn.Conv2d(ch, 1, kernel_size=1, bias=False),
                                 nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class FEB0(nn.Module):
    def __init__(self):
        super(FEB0, self).__init__()
        self.layer_c4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_d6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_d9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_up12 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_r1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_c13 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_up15 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_r2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_c16 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c17 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_ups1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_rs1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_cus1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_cus2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
    def forward(self, x):
        cat1 = x
        scat1 = cat1
        c4 = self.layer_c4(cat1)
        c5 = self.layer_c5(c4)
        sc5 = c5
        c5 = c5 + scat1
        d6 = self.layer_d6(c5)
        c7 = self.layer_c7(d6)
        c8 = self.layer_c8(c7)
        sc8 = c8
        c8 = c8 + d6
        d9 = self.layer_d9(c8)
        c10 = self.layer_c10(d9)
        c11 = self.layer_c11(c10)
        c11 = c11 + d9
        up12 = self.layer_up12(c11)
        cat2 = torch.cat([up12, sc8], dim=1)
        r1 = self.layer_r1(cat2)
        sr1 = r1
        c13 = self.layer_c13(r1)
        c14 = self.layer_c14(c13)
        c14 = c14 + sr1
        up15 = self.layer_up15(c14)
        ups1 = self.layer_ups1(c8)
        catus1 = torch.cat([sc5, ups1], dim=1)
        rs1 = self.layer_rs1(catus1)
        cus1 = self.layer_cus1(rs1)
        cus2 = self.layer_cus2(cus1)
        cus2 = cus2 + rs1
        cat3 = torch.cat([up15, sc5, cus2], dim=1)
        r2 = self.layer_r2(cat3)
        sr2 = r2
        c16 = self.layer_c16(r2)
        c17 = self.layer_c17(c16)
        c17 = c17 + sr2
        return c5, cus2, c17


class FEB1(nn.Module):
    def __init__(self):
        super(FEB1, self).__init__()
        self.layer_c1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU()
        )
        self.layer_c4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_d6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_d9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_up12 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_r1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_c13 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_up15 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_r2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_c16 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c17 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_ups1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_rs1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_cus1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_cus2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
    def forward(self, x):
        cat1 = self.layer_c1(x)
        scat1 = cat1
        c4 = self.layer_c4(cat1)
        c5 = self.layer_c5(c4)
        sc5 = c5
        c5 = c5 + scat1
        d6 = self.layer_d6(c5)
        c7 = self.layer_c7(d6)
        c8 = self.layer_c8(c7)
        sc8 = c8
        c8 = c8 + d6
        d9 = self.layer_d9(c8)
        c10 = self.layer_c10(d9)
        c11 = self.layer_c11(c10)
        c11 = c11 + d9
        up12 = self.layer_up12(c11)
        cat2 = torch.cat([up12, sc8], dim=1)
        r1 = self.layer_r1(cat2)
        sr1 = r1
        c13 = self.layer_c13(r1)
        c14 = self.layer_c14(c13)
        c14 = c14 + sr1
        up15 = self.layer_up15(c14)
        ups1 = self.layer_ups1(c8)
        catus1 = torch.cat([sc5, ups1], dim=1)
        rs1 = self.layer_rs1(catus1)
        cus1 = self.layer_cus1(rs1)
        cus2 = self.layer_cus2(cus1)
        cus2 = cus2 + rs1
        cat3 = torch.cat([up15, sc5, cus2], dim=1)
        r2 = self.layer_r2(cat3)
        sr2 = r2
        c16 = self.layer_c16(r2)
        c17 = self.layer_c17(c16)
        c17 = c17 + sr2
        return c5, cus2, c17


class FEB0(nn.Module):
    def __init__(self):
        super(FEB0, self).__init__()
        self.layer_c4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_d6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_d9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_up12 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_r1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_c13 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_up15 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_r2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_c16 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c17 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_ups1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_rs1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_cus1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_cus2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
    def forward(self, x):
        cat1 = x
        scat1 = cat1
        c4 = self.layer_c4(cat1)
        c5 = self.layer_c5(c4)
        sc5 = c5
        c5 = c5 + scat1
        d6 = self.layer_d6(c5)
        c7 = self.layer_c7(d6)
        c8 = self.layer_c8(c7)
        sc8 = c8
        c8 = c8 + d6
        d9 = self.layer_d9(c8)
        c10 = self.layer_c10(d9)
        c11 = self.layer_c11(c10)
        c11 = c11 + d9
        up12 = self.layer_up12(c11)
        cat2 = torch.cat([up12, sc8], dim=1)
        r1 = self.layer_r1(cat2)
        sr1 = r1
        c13 = self.layer_c13(r1)
        c14 = self.layer_c14(c13)
        c14 = c14 + sr1
        up15 = self.layer_up15(c14)
        ups1 = self.layer_ups1(c8)
        catus1 = torch.cat([sc5, ups1], dim=1)
        rs1 = self.layer_rs1(catus1)
        cus1 = self.layer_cus1(rs1)
        cus2 = self.layer_cus2(cus1)
        cus2 = cus2 + rs1
        cat3 = torch.cat([up15, sc5, cus2], dim=1)
        r2 = self.layer_r2(cat3)
        sr2 = r2
        c16 = self.layer_c16(r2)
        c17 = self.layer_c17(c16)
        c17 = c17 + sr2
        return c5, cus2, c17


class FEB1(nn.Module):
    def __init__(self):
        super(FEB1, self).__init__()
        self.layer_c1 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PReLU()
        )
        self.layer_c4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_d6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c7 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_d9 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c10 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c11 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_up12 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_r1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_c13 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c14 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_up15 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_r2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_c16 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c17 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_ups1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_rs1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_cus1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_cus2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
    def forward(self, x):
        cat1 = self.layer_c1(x)
        scat1 = cat1
        c4 = self.layer_c4(cat1)
        c5 = self.layer_c5(c4)
        sc5 = c5
        c5 = c5 + scat1
        d6 = self.layer_d6(c5)
        c7 = self.layer_c7(d6)
        c8 = self.layer_c8(c7)
        sc8 = c8
        c8 = c8 + d6
        d9 = self.layer_d9(c8)
        c10 = self.layer_c10(d9)
        c11 = self.layer_c11(c10)
        c11 = c11 + d9
        up12 = self.layer_up12(c11)
        cat2 = torch.cat([up12, sc8], dim=1)
        r1 = self.layer_r1(cat2)
        sr1 = r1
        c13 = self.layer_c13(r1)
        c14 = self.layer_c14(c13)
        c14 = c14 + sr1
        up15 = self.layer_up15(c14)
        ups1 = self.layer_ups1(c8)
        catus1 = torch.cat([sc5, ups1], dim=1)
        rs1 = self.layer_rs1(catus1)
        cus1 = self.layer_cus1(rs1)
        cus2 = self.layer_cus2(cus1)
        cus2 = cus2 + rs1
        cat3 = torch.cat([up15, sc5, cus2], dim=1)
        r2 = self.layer_r2(cat3)
        sr2 = r2
        c16 = self.layer_c16(r2)
        c17 = self.layer_c17(c16)
        c17 = c17 + sr2
        return c5, cus2, c17


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer_lm = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=4, stride=4, output_padding=0)
        )
        self.layer_hp = nn.Sequential(
            HighpassConv(1)
        )
        self.layer_p1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_m1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_21 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.layer_22 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.layer_23 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.layer_24 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.layer_3 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, bias=False),
        )
        self.layer_scse = nn.Sequential(
            SCSE(64)
        )
        self.layer_4 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.layer_dp = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
        )
        self.layer_dm = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
        )
        self.FEB1 = nn.Sequential(
            FEB0(),
        )
        self.FEB2 = nn.Sequential(
            FEB0(),
        )
        self.FEB3 = nn.Sequential(
            FEB0(),
        )
        self.FEB4 = nn.Sequential(
            FEB0(),
        )
        self.layer_up18 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_r3 = nn.Sequential(
            nn.Conv2d(in_channels=352, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_c19 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c20 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_c21 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=4, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.Tanh(),
        )

        self.layer_ups1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_rs1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_cus1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_cus2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_ups2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_rs2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_cus3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_cus4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_ups3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, output_padding=0),
            nn.PReLU(),
        )
        self.layer_rs3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
        )
        self.layer_cus5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
        self.layer_cus6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.PReLU(),
        )
    def forward(self, ms, pan):
        ms = self.layer_lm(ms)
        pan1 = self.layer_p1(pan)
        rpan1 = pan1
        pan21 = self.layer_21(pan1)
        pan22 = self.layer_22(pan1)
        pan23 = self.layer_23(pan1)
        pan24 = self.layer_24(pan1)
        pan2 = torch.cat([pan21, pan22, pan23, pan24], dim=1)
        pan3 = self.layer_3(pan2)
        sep = self.layer_scse(pan3)
        rpan3 = sep + rpan1
        pan4 = self.layer_4(rpan3)
        dp = self.layer_dp(pan4)
        ms1 = self.layer_m1(ms)
        rms1 = ms1
        ms21 = self.layer_21(ms1)
        ms22 = self.layer_22(ms1)
        ms23 = self.layer_23(ms1)
        ms24 = self.layer_24(ms1)
        ms2 = torch.cat([ms21, ms22, ms23, ms24], dim=1)
        ms3 = self.layer_3(ms2)
        sem = self.layer_scse(ms3)
        rms3 = sem + rms1
        ms4 = self.layer_4(rms3)
        dm = self.layer_dm(ms4)
        cat1 = torch.cat([dm, dp], dim=1)
        F1c5, F1cus2, F1 = self.FEB1(cat1)
        cat5 = torch.cat([cat1, F1], dim=1)
        F2c5, F2cus2, F2 = self.FEB2(cat5)
        cat6 = torch.cat([cat1, F2], dim=1)
        F3c5, F3cus2, F3 = self.FEB3(cat6)
        cat7 = torch.cat([cat1, F3], dim=1)
        F4c5, F4cus2, F4 = self.FEB4(cat7)

        F1up18 = self.layer_up18(F1)
        F1ups2 = self.layer_ups2(F1c5)
        F1catus2 = torch.cat([pan4, ms4, F1ups2], dim=1)
        F1rs2 = self.layer_rs2(F1catus2)
        F1cus3 = self.layer_cus3(F1rs2)
        F1cus4 = self.layer_cus4(F1cus3)
        F1cus4 = F1cus4 + F1rs2
        F1ups3 = self.layer_ups3(F1cus2)
        F1catus3 = torch.cat([pan4, ms4, F1cus4, F1ups3], dim=1)
        F1rs3 = self.layer_rs3(F1catus3)
        F1cus5 = self.layer_cus5(F1rs3)
        F1cus6 = self.layer_cus6(F1cus5)
        F1cus6 = F1cus6 + F1rs3
        F1cat4 = torch.cat([F1up18, pan4, ms4, F1cus4, F1cus6], dim=1)
        F1r3 = self.layer_r3(F1cat4)
        F1sr3 = F1r3
        F1c19 = self.layer_c19(F1r3)
        F1c20 = self.layer_c20(F1c19)
        F1c20 = F1c20 + F1sr3
        F1 = self.layer_c21(F1c20)

        F2up18 = self.layer_up18(F2)
        F2ups2 = self.layer_ups2(F2c5)
        F2catus2 = torch.cat([pan4, ms4, F2ups2], dim=1)
        F2rs2 = self.layer_rs2(F2catus2)
        F2cus3 = self.layer_cus3(F2rs2)
        F2cus4 = self.layer_cus4(F2cus3)
        F2cus4 = F2cus4 + F2rs2
        F2ups3 = self.layer_ups3(F2cus2)
        F2catus3 = torch.cat([pan4, ms4, F2cus4, F2ups3], dim=1)
        F2rs3 = self.layer_rs3(F2catus3)
        F2cus5 = self.layer_cus5(F2rs3)
        F2cus6 = self.layer_cus6(F2cus5)
        F2cus6 = F2cus6 + F2rs3
        F2cat4 = torch.cat([F2up18, pan4, ms4, F2cus4, F2cus6], dim=1)
        F2r3 = self.layer_r3(F2cat4)
        F2sr3 = F2r3
        F2c19 = self.layer_c19(F2r3)
        F2c20 = self.layer_c20(F2c19)
        F2c20 = F2c20 + F2sr3
        F2 = self.layer_c21(F2c20)

        F3up18 = self.layer_up18(F3)
        F3ups2 = self.layer_ups2(F3c5)
        F3catus2 = torch.cat([pan4, ms4, F3ups2], dim=1)
        F3rs2 = self.layer_rs2(F3catus2)
        F3cus3 = self.layer_cus3(F3rs2)
        F3cus4 = self.layer_cus4(F3cus3)
        F3cus4 = F3cus4 + F3rs2
        F3ups3 = self.layer_ups3(F3cus2)
        F3catus3 = torch.cat([pan4, ms4, F3cus4, F3ups3], dim=1)
        F3rs3 = self.layer_rs3(F3catus3)
        F3cus5 = self.layer_cus5(F3rs3)
        F3cus6 = self.layer_cus6(F3cus5)
        F3cus6 = F3cus6 + F3rs3
        F3cat4 = torch.cat([F3up18, pan4, ms4, F3cus4, F3cus6], dim=1)
        F3r3 = self.layer_r3(F3cat4)
        F3sr3 = F3r3
        F3c19 = self.layer_c19(F3r3)
        F3c20 = self.layer_c20(F3c19)
        F3c20 = F3c20 + F3sr3
        F3 = self.layer_c21(F3c20)

        F4up18 = self.layer_up18(F4)
        F4ups2 = self.layer_ups2(F4c5)
        F4catus2 = torch.cat([pan4, ms4, F4ups2], dim=1)
        F4rs2 = self.layer_rs2(F4catus2)
        F4cus3 = self.layer_cus3(F4rs2)
        F4cus4 = self.layer_cus4(F4cus3)
        F4cus4 = F4cus4 + F4rs2
        F4ups3 = self.layer_ups3(F4cus2)
        F4catus3 = torch.cat([pan4, ms4, F4cus4, F4ups3], dim=1)
        F4rs3 = self.layer_rs3(F4catus3)
        F4cus5 = self.layer_cus5(F4rs3)
        F4cus6 = self.layer_cus6(F4cus5)
        F4cus6 = F4cus6 + F4rs3
        F4cat4 = torch.cat([F4up18, pan4, ms4, F4cus4, F4cus6], dim=1)
        F4r3 = self.layer_r3(F4cat4)
        F4sr3 = F4r3
        F4c19 = self.layer_c19(F4r3)
        F4c20 = self.layer_c20(F4c19)
        F4c20 = F4c20 + F4sr3
        F4 = self.layer_c21(F4c20)
        return F1, F2, F3, F4

if __name__ == '__main__':
    model = Net
    model = torch.load('./models/model-MYNet.pth')
    model.to(device)
    model.eval()
    test_data = './Qiuck_data.mat'
    data = sio.loadmat(test_data)
    ms = data['ms'][...]  # MS image
    ms = np.array(ms, dtype=np.float32) / 2047.

    lms = data['lms'][...]  # up-sampled LRMS image
    lms = np.array(lms, dtype=np.float32) / 2047.

    pan = data['pan'][...]  # PAN image
    pan = np.array(pan, dtype=np.float32) / 2047.

    lms = lms[np.newaxis, :, :, :]
    pan = pan[np.newaxis, :, :, np.newaxis]
    pan = np.transpose(pan, (0, 3, 1, 2))
    ms = ms[np.newaxis, :, :, :]
    ms = np.transpose(ms, (0, 3, 1, 2))
    lms = np.transpose(lms, (0, 3, 1, 2))

    ms = Variable(torch.from_numpy(ms)).type(dtype)
    pan = Variable(torch.from_numpy(pan)).type(dtype)
    lms = Variable(torch.from_numpy(lms)).type(dtype)
    ms = ms.to(device)
    pan = pan.to(device)
    lms = lms.to(device)

    F1, F2, F3, TF = model(ms, pan)
    TF = TF + lms
    TF = TF.data.cpu().numpy()
    TF = np.clip(TF, 0, 1)
    TF = np.transpose(TF, (2, 3, 1, 0))
    TF = np.squeeze(TF)
    print(TF.shape)
    sio.savemat('./result/MYNET.mat', {'result': TF})
