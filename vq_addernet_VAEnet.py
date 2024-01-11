# 2020.01.10-Replaced conv with adder
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
from __future__ import print_function
import abc

import adder
import torch.nn as nn
import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from nearest_embed import NearestEmbed, NearestEmbedEMA
from vars import *


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return adder.adder2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride = stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Conv2d(64 * block.expansion, num_classes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
         
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                adder.adder2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(inplanes = self.inplanes, planes = planes, stride = stride, downsample = downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes = self.inplanes, planes = planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn2(x)

        return x.view(x.size(0), -1)


def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()
        self.quant_rb = torch.quantization.QuantStub()
        self.dequant_rb = torch.quantization.DeQuantStub()
        self.dequant_cnv = torch.quantization.DeQuantStub()
        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),

            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]

        # layers = [
        #     nn.ReLU(),
        #     adder.adder2d(in_channels, mid_channels,
        #             kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     adder.adder2d(mid_channels, out_channels,
        #             kernel_size=1, stride=1, padding=0)        
        # ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        fp32x=self.dequant_rb(x)
        cnvtemp=self.convs(x)
        fp32cnvtemp=self.dequant_cnv(cnvtemp)
        fp32output = fp32x + fp32cnvtemp
        output = self.quant_rb(fp32output)
        return output
        # if(x.is_quantized):
        #     temp=self.dequant3(x)
        #     cnvdeq=self.dequant3(self.convs(x))
        #     output = temp + cnvdeq
        #     # temp = x.dequantize()
        #     # cnvdeq=self.convs(x).dequantize()
        #     # temp = temp + cnvdeq
        #     # min_val, max_val = temp.min(), temp.max()
        #     # scale = (max_val - min_val) / (255 - 0)
        #     # zero_point =0 - min_val / scale
        #     # # scale, zero_point = 0.004, 0
        #     # dtype = torch.quint8
        #     # temp = torch.quantize_per_tensor(temp, scale, zero_point, dtype)
        #     output=self.quant3(output)
        #     return output
        
        # else:
        #     return x + self.convs(x)
        

class adderResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(adderResBlock, self).__init__()
        self.quant_rb = torch.quantization.QuantStub()
        self.dequant_rb = torch.quantization.DeQuantStub()
        self.dequant_cnv = torch.quantization.DeQuantStub()
        if mid_channels is None:
            mid_channels = out_channels

        # layers = [
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels, mid_channels,
        #               kernel_size=3, stride=1, padding=1),

        #     nn.ReLU(),
        #     nn.Conv2d(mid_channels, out_channels,
        #               kernel_size=1, stride=1, padding=0)
        # ]

        layers = [
            nn.ReLU(),
            adder.adder2d(in_channels, mid_channels,
                    kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            adder.adder2d(mid_channels, out_channels,
                    kernel_size=1, stride=1, padding=0)        
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        # if(x.is_quantized):
        tempx=self.dequant_rb(x)
        cnvtemp=self.convs(tempx)
        cnvtemp=self.dequant_cnv(cnvtemp)
        output = tempx + cnvtemp
        output = self.quant_rb(output)
        return output
     
        

class VQ_CVAE(nn.Module):
    def __init__(self, d, k, bn=True, vq_coef=1, commit_coef=0.5, num_channels=n_channels, **kwargs):
        super(VQ_CVAE, self).__init__()
        self.quant1 = torch.quantization.QuantStub()
        self.dequant1 = torch.quantization.DeQuantStub()
        self.quant2 = torch.quantization.QuantStub()
        self.dequant2 = torch.quantization.DeQuantStub()

        # self.conve1 = nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1)
        self.conve1 = adder.adder2d(num_channels, d, kernel_size=4, stride=2, padding=1)#采用adder网络
        self.bne1 = nn.BatchNorm2d(d)
        self.relue1 =nn.ReLU(inplace=True)
        # self.conve2 = nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1)
        self.conve2 = adder.adder2d(d, d, kernel_size=4, stride=2, padding=1)#采用adder网络
        self.bne2= nn.BatchNorm2d(d)
        self.relue2 =nn.ReLU(inplace=True)
        self.resblocke1 = adderResBlock(d, d, bn=bn)
        self.bne3 = nn.BatchNorm2d(d)
        self.resblocke2 = adderResBlock(d, d, bn=bn)
        self.bne4 = nn.BatchNorm2d(d)


        # self.convd1 = adder.adder2d(d, d, kernel_size=3, stride=1, padding=1)#采用adder网络替代普通的卷积层
        # self.convd2 = adder.adder2d(d, num_channels, kernel_size=3, stride=1, padding=1)#采用adder网络替代普通的卷积层
        self.convd1 = nn.Conv2d(d, d, kernel_size=3, stride=1, padding=1)#采用普通的卷积层解码
        self.convd2 = nn.Conv2d(d, num_channels, kernel_size=3, stride=1, padding=1)#采用普通的卷积层解码
        self.upsample1=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample2=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.resblockd1 = ResBlock(d, d)
        self.bnd1 = nn.BatchNorm2d(d)
        self.resblockd2 = ResBlock(d, d)
        # self.convtranspose1=nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1)
        self.bnd2 = nn.BatchNorm2d(d)
        self.relud1 = nn.ReLU(inplace=True)
        # self.convtranspose2=nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1)
        
        self.d = d
        self.emb = NearestEmbed(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        # self.encoder[-1].weight.detach().fill_(1 / 40)
        self.bne4.weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

        
    def forward(self, x):
        x = self.quant1(x)
        # 以下是编码器
        # xforadd1 = self.dequant(x0)

        x1 =self.conve1(x)
        # x1 =self.quant1(x1)
        
        x =self.bne1(x1)
        x =self.relue1(x)

        # x =self.dequant2(x)
        x2 =self.conve2(x)
        # x2 =self.quant2(x2)

        x =self.bne2(x2)
        x =self.relue2(x)
        x =self.resblocke1(x)
        x =self.bne3(x)
        x =self.resblocke2(x)
        z_e =self.bne4(x)
        


        #以下是量化器
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        

        z_qq = self.quant2(z_q)
        # #以下是解码器
        y =self.resblockd1(z_qq)
        y =self.bnd1(y)
        y =self.resblockd2(y)
        
        # y=self.convtranspose1(y)#这里是转置卷积层
        y =self.upsample1(y)#这里是上采样
        y =self.convd1(y)#这里是普通的卷积层
        y =self.bnd2(y)
        y =self.relud1(y)

        # y=self.convtranspose2(y)#这里是转置卷积层
        y =self.upsample2(y)#这里是上采样
        y =self.convd2(y)#这里是普通的卷积层
        y=torch.tanh(y)
        y =self.dequant2(y)
        return y, z_e, emb, argmin


    def sample(self, size):
        sample = torch.randn(size, self.d, self.f,
                             self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb):
        self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach())**2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e)**2, 2, 1))

        return self.mse + self.vq_coef*self.vq_loss + self.commit_coef*self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):

        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)


def convVAE(**kwargs):
    model=VQ_CVAE(d=emb_dim,k=emb_num,commit_coef=commit_beta,**kwargs)
    # model.half()
    return model