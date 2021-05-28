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
    def __init__(self, channels=1):
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
            nn.PReLU(),
            nn.Conv2d(ch//re, ch, kernel_size=1, bias=False),
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
                                 nn.PReLU(),
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
        # self.layer_21 = nn.Sequential(
        #     nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        # )
        # self.layer_22 = nn.Sequential(
        #     nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        # )
        # self.layer_23 = nn.Sequential(
        #     nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        # )
        # self.layer_24 = nn.Sequential(
        #     nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.PReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        # )
        self.layer_21 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1, bias=False, dilation=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False, dilation=1),
        )
        self.layer_22 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
        )
        self.layer_23 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=3, bias=False, dilation=3),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=3, bias=False, dilation=3),
        )
        self.layer_24 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=4, bias=False, dilation=4),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=4, bias=False, dilation=4),
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
            FEB1(),
        )
        self.FEB3 = nn.Sequential(
            FEB1(),
        )
        self.FEB4 = nn.Sequential(
            FEB1(),
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

        # f2pan1 = self.layer_p1(pan)
        # f2rpan1 = f2pan1
        # f2pan21 = self.layer_21(f2pan1)
        # f2pan22 = self.layer_22(f2pan1)
        # f2pan23 = self.layer_23(f2pan1)
        # f2pan24 = self.layer_24(f2pan1)
        # f2pan2 = torch.cat([f2pan21, f2pan22, f2pan23, f2pan24], dim=1)
        # f2pan3 = self.layer_3(f2pan2)
        # f2sep = self.layer_scse(f2pan3)
        # f2rpan3 = f2sep + f2rpan1
        # f2pan4 = self.layer_4(f2rpan3)
        # f2dp = self.layer_dp(f2pan4)
        # f2ms1 = self.layer_m1(ms)
        # f2rms1 = f2ms1
        # f2ms21 = self.layer_21(f2ms1)
        # f2ms22 = self.layer_22(f2ms1)
        # f2ms23 = self.layer_23(f2ms1)
        # f2ms24 = self.layer_24(f2ms1)
        # f2ms2 = torch.cat([f2ms21, f2ms22, f2ms23, f2ms24], dim=1)
        # f2ms3 = self.layer_3(f2ms2)
        # f2sem = self.layer_scse(f2ms3)
        # f2rms3 = f2sem + f2rms1
        # f2ms4 = self.layer_4(f2rms3)
        # f2dm = self.layer_dm(f2ms4)
        # catf2 = torch.cat([f2dm, f2dp], dim=1)
        # cat5 = torch.cat([catf2, F1], dim=1)
        # F2c5, F2cus2, F2 = self.FEB2(cat5)
        #
        # f3pan1 = self.layer_p1(pan)
        # f3rpan1 = f3pan1
        # f3pan21 = self.layer_21(f3pan1)
        # f3pan22 = self.layer_22(f3pan1)
        # f3pan23 = self.layer_23(f3pan1)
        # f3pan24 = self.layer_24(f3pan1)
        # f3pan2 = torch.cat([f3pan21, f3pan22, f3pan23, f3pan24], dim=1)
        # f3pan3 = self.layer_3(f3pan2)
        # f3sep = self.layer_scse(f3pan3)
        # f3rpan3 = f3sep + f3rpan1
        # f3pan4 = self.layer_4(f3rpan3)
        # f3dp = self.layer_dp(f3pan4)
        # f3ms1 = self.layer_m1(ms)
        # f3rms1 = f3ms1
        # f3ms21 = self.layer_21(f3ms1)
        # f3ms22 = self.layer_22(f3ms1)
        # f3ms23 = self.layer_23(f3ms1)
        # f3ms24 = self.layer_24(f3ms1)
        # f3ms2 = torch.cat([f3ms21, f3ms22, f3ms23, f3ms24], dim=1)
        # f3ms3 = self.layer_3(f3ms2)
        # f3sem = self.layer_scse(f3ms3)
        # f3rms3 = f3sem + f3rms1
        # f3ms4 = self.layer_4(f3rms3)
        # f3dm = self.layer_dm(f3ms4)
        # catf3 = torch.cat([f3dm, f3dp], dim=1)
        # cat6 = torch.cat([catf3, F2], dim=1)
        # F3c5, F3cus2, F3 = self.FEB3(cat6)
        #
        # f4pan1 = self.layer_p1(pan)
        # f4rpan1 = f4pan1
        # f4pan21 = self.layer_21(f4pan1)
        # f4pan22 = self.layer_22(f4pan1)
        # f4pan23 = self.layer_23(f4pan1)
        # f4pan24 = self.layer_24(f4pan1)
        # f4pan2 = torch.cat([f4pan21, f4pan22, f4pan23, f4pan24], dim=1)
        # f4pan3 = self.layer_3(f4pan2)
        # f4sep = self.layer_scse(f4pan3)
        # f4rpan3 = f4sep + f4rpan1
        # f4pan4 = self.layer_4(f4rpan3)
        # f4dp = self.layer_dp(f4pan4)
        # f4ms1 = self.layer_m1(ms)
        # f4rms1 = f4ms1
        # f4ms21 = self.layer_21(f4ms1)
        # f4ms22 = self.layer_22(f4ms1)
        # f4ms23 = self.layer_23(f4ms1)
        # f4ms24 = self.layer_24(f4ms1)
        # f4ms2 = torch.cat([f4ms21, f4ms22, f4ms23, f4ms24], dim=1)
        # f4ms3 = self.layer_3(f4ms2)
        # f4sem = self.layer_scse(f4ms3)
        # f4rms3 = f4sem + f4rms1
        # f4ms4 = self.layer_4(f4rms3)
        # f4dm = self.layer_dm(f4ms4)
        # catf4 = torch.cat([f4dm, f4dp], dim=1)
        # cat7 = torch.cat([catf4, F3], dim=1)
        # F4c5, F4cus2, F4 = self.FEB4(cat7)
        #
        # F1up18 = self.layer_up18(F1)
        # F1ups2 = self.layer_ups2(F1c5)
        # F1catus2 = torch.cat([pan4, ms4, F1ups2], dim=1)
        # F1rs2 = self.layer_rs2(F1catus2)
        # F1cus3 = self.layer_cus3(F1rs2)
        # F1cus4 = self.layer_cus4(F1cus3)
        # F1cus4 = F1cus4 + F1rs2
        # F1ups3 = self.layer_ups3(F4cus2)
        # F1catus3 = torch.cat([pan4, ms4, F1cus4, F1ups3], dim=1)
        # F1rs3 = self.layer_rs3(F1catus3)
        # F1cus5 = self.layer_cus5(F1rs3)
        # F1cus6 = self.layer_cus6(F1cus5)
        # F1cus6 = F1cus6 + F1rs3
        # F1cat4 = torch.cat([F1up18, pan4, ms4, F1cus4, F1cus6], dim=1)
        # F1r3 = self.layer_r3(F1cat4)
        # F1sr3 = F1r3
        # F1c19 = self.layer_c19(F1r3)
        # F1c20 = self.layer_c20(F1c19)
        # F1c20 = F1c20 + F1sr3
        # F1 = self.layer_c21(F1c20)
        #
        # F2up18 = self.layer_up18(F2)
        # F2ups2 = self.layer_ups2(F2c5)
        # F2catus2 = torch.cat([f2pan4, f2ms4, F2ups2], dim=1)
        # F2rs2 = self.layer_rs2(F2catus2)
        # F2cus3 = self.layer_cus3(F2rs2)
        # F2cus4 = self.layer_cus4(F2cus3)
        # F2cus4 = F2cus4 + F2rs2
        # F2ups3 = self.layer_ups3(F2cus2)
        # F2catus3 = torch.cat([f2pan4, f2ms4, F2cus4, F2ups3], dim=1)
        # F2rs3 = self.layer_rs3(F2catus3)
        # F2cus5 = self.layer_cus5(F2rs3)
        # F2cus6 = self.layer_cus6(F2cus5)
        # F2cus6 = F2cus6 + F2rs3
        # F2cat4 = torch.cat([F2up18, f2pan4, f2ms4, F2cus4, F2cus6], dim=1)
        # F2r3 = self.layer_r3(F2cat4)
        # F2sr3 = F2r3
        # F2c19 = self.layer_c19(F2r3)
        # F2c20 = self.layer_c20(F2c19)
        # F2c20 = F2c20 + F2sr3
        # F2 = self.layer_c21(F2c20)
        #
        # F3up18 = self.layer_up18(F3)
        # F3ups2 = self.layer_ups2(F3c5)
        # F3catus2 = torch.cat([f3pan4, f3ms4, F3ups2], dim=1)
        # F3rs2 = self.layer_rs2(F3catus2)
        # F3cus3 = self.layer_cus3(F3rs2)
        # F3cus4 = self.layer_cus4(F3cus3)
        # F3cus4 = F3cus4 + F3rs2
        # F3ups3 = self.layer_ups3(F3cus2)
        # F3catus3 = torch.cat([f3pan4, f3ms4, F3cus4, F3ups3], dim=1)
        # F3rs3 = self.layer_rs3(F3catus3)
        # F3cus5 = self.layer_cus5(F3rs3)
        # F3cus6 = self.layer_cus6(F3cus5)
        # F3cus6 = F3cus6 + F3rs3
        # F3cat4 = torch.cat([F3up18, f3pan4, f3ms4, F3cus4, F3cus6], dim=1)
        # F3r3 = self.layer_r3(F3cat4)
        # F3sr3 = F3r3
        # F3c19 = self.layer_c19(F3r3)
        # F3c20 = self.layer_c20(F3c19)
        # F3c20 = F3c20 + F3sr3
        # F3 = self.layer_c21(F3c20)
        #
        # F4up18 = self.layer_up18(F4)
        # F4ups2 = self.layer_ups2(F4c5)
        # F4catus2 = torch.cat([f4pan4, f4ms4, F4ups2], dim=1)
        # F4rs2 = self.layer_rs2(F4catus2)
        # F4cus3 = self.layer_cus3(F4rs2)
        # F4cus4 = self.layer_cus4(F4cus3)
        # F4cus4 = F4cus4 + F4rs2
        # F4ups3 = self.layer_ups3(F4cus2)
        # F4catus3 = torch.cat([f4pan4, f4ms4, F4cus4, F4ups3], dim=1)
        # F4rs3 = self.layer_rs3(F4catus3)
        # F4cus5 = self.layer_cus5(F4rs3)
        # F4cus6 = self.layer_cus6(F4cus5)
        # F4cus6 = F4cus6 + F4rs3
        # F4cat4 = torch.cat([F4up18, f4pan4, f4ms4, F4cus4, F4cus6], dim=1)
        # F4r3 = self.layer_r3(F4cat4)
        # F4sr3 = F4r3
        # F4c19 = self.layer_c19(F4r3)
        # F4c20 = self.layer_c20(F4c19)
        # F4c20 = F4c20 + F4sr3
        # F4 = self.layer_c21(F4c20)

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
        F1ups3 = self.layer_ups3(F4cus2)
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


# get training patches
def get_batch(train_data, bs):
    gt = train_data['gt'][...]  ## ground truth N*H*W*C
    gt = np.transpose(gt, (0, 3, 1, 2))
    pan = train_data['pan'][...]  #### Pan image N*H*W
    pan = pan[:, :, :, np.newaxis]
    pan = np.transpose(pan, (0, 3, 1, 2))
    ms_lr = train_data['ms'][...]  ### low resolution MS image
    ms_lr = np.transpose(ms_lr, (0, 3, 1, 2))
    lms = train_data['lms'][...]  #### MS image interpolation to Pan scale
    lms = np.transpose(lms, (0, 3, 1, 2))

    gt = np.array(gt, dtype=np.float32) / 2047.  ### normalization, WorldView L = 11
    pan = np.array(pan, dtype=np.float32) / 2047.
    ms_lr = np.array(ms_lr, dtype=np.float32) / 2047.
    lms = np.array(lms, dtype=np.float32) / 2047.

    N = gt.shape[0]
    batch_index = np.random.randint(0, N, size=bs)

    gt_batch = gt[batch_index, :, :, :]
    pan_batch = pan[batch_index, :, :]
    ms_lr_batch = ms_lr[batch_index, :, :, :]
    lms_batch = lms[batch_index, :, :, :]

    return gt_batch, lms_batch, pan_batch, ms_lr_batch


if __name__ == '__main__':
    train_batch_size = 64  # training batch size
    test_batch_size = 16  # validation batch size
    image_size = 64  # patch size
    checkpoint_iter = 0
    iterations = 40100  # total number of iterations to use.
    model_directory = './models'  # directory to save trained model to.
    train_data_name = './training_data/train.mat'  # training data
    test_data_name = './training_data/validation.mat'  # validation data
    LR = 0.0008
    modelType = 'IDM'

    ############## loading data
    train_data = sio.loadmat(train_data_name)
    test_data = sio.loadmat(test_data_name)
    net = Net()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 是否有多个GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        # PyTorch默认只使用一个GPU，使用DataParallel来实现使用多个GPU
        model = nn.DataParallel(net)
    else:
        print("only one")
    # 将模型放入GPU
    net.to(device)
    # L2 = torch.nn.MSELoss().type(dtype)
    # L2 = L2.to(device)
    L1 = torch.nn.L1Loss().type(dtype)
    L1 = L1.to(device)
    if checkpoint_iter != 0:
        net = torch.load('models/model-MYNet1-32000.pth')

    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.8)

    for i in range(iterations):
        ############## placeholder for training
        gt, lms, pan, ms = get_batch(train_data, test_batch_size)
        optimizer.zero_grad()
        ms = Variable(torch.from_numpy(ms)).type(dtype)
        pan = Variable(torch.from_numpy(pan)).type(dtype)
        gt = Variable(torch.from_numpy(gt)).type(dtype)
        lms = Variable(torch.from_numpy(lms)).type(dtype)
        ms = ms.to(device)
        pan = pan.to(device)
        gt = gt.to(device)
        lms = lms.to(device)
        F1, F2, F3, F4 = net(ms, pan)
        F1 = F1 + lms
        F2 = F2 + lms
        F3 = F3 + lms
        F4 = F4 + lms
        # l2 = L2(F1, gt) + L2(F2, gt) + L2(F3, gt) + L2(F4, gt)
        # loss = l2 / 4
        l1 = L1(F1, gt) + L1(F2, gt) + L1(F3, gt) + L1(F4, gt)
        loss = l1 / 4
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 10 == 0:
            print('Iteration= %d   L= %.8f' % (i, loss.item()))
            if i % 1000 == 0 and i != 0:
                te_gt, te_lms, te_pan, te_ms = get_batch(test_data, test_batch_size)
                te_ms = Variable(torch.from_numpy(te_ms)).type(dtype)
                te_pan = Variable(torch.from_numpy(te_pan)).type(dtype)
                te_gt = Variable(torch.from_numpy(te_gt)).type(dtype)
                te_lms = Variable(torch.from_numpy(te_lms)).type(dtype)
                te_F1, te_F2, te_F3, te_F4 = net(te_ms, te_pan)
                F1 = te_F1 + te_lms
                F2 = te_F2 + te_lms
                F3 = te_F3 + te_lms
                F4 = te_F4 + te_lms
                te_l1 = L1(F1, te_gt) + L1(F2, te_gt) + L1(F3, te_gt) + L1(F4, te_gt)
                te_loss = te_l1 / 4
                netOutput_np = F4.cpu().data.numpy()
                netLabel_np = te_gt.cpu().data.numpy()
                psnrValue = psnr(netLabel_np, netOutput_np)
                print('Iteration= %d   validation_MAE= %.8f' % (i, te_loss.item()))
                print('psnr %.2f' % (psnrValue))
            if i % 8000 == 0 and i != 0:
                model_name = os.path.join('models/model-MYNet-%d.pth' % i)
                # model_name = os.path.join('models/model-MYNet-40000.pth')
                torch.save(net, model_name)
