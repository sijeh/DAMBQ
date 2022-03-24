'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
import math
from models.quant_layer import *
from collections import OrderedDict
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def Quantconv3x3(in_planes, out_planes, stride=1):
    " 3x3 quantized convolution with padding "
    return QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, float=False):
        super(BasicBlock, self).__init__()
        if float:
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv1 = Quantconv3x3(inplanes, planes, stride)
            self.conv2 = Quantconv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.ReLU(inplace=True)  # nn.PReLU(num_parameters=planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = nn.ReLU(inplace=True)  # nn.PReLU(num_parameters=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        '''
        with torch.no_grad():
            std_out = out.std(dim=(0,2,3))
            std = out.std()
            mean_out = out.mean()
            print('after BN->mean:%.3f\t std_mean:%.3f\t std_min:%.3f\t std_max:%.3f\t std:%.3f'%(mean_out.item(),std_out.mean().item(),std_out.min().item(),std_out.max().item(),std.item()))
        #print('std:',std_out.mean().item(),'mean:',mean_out.item(),'min:',std_out.min().item(),'max:',std_out.max().item())
        '''
        out = self.prelu1(out)
        '''
        with torch.no_grad():
            std_out = out.std(dim=(0,2,3))
            std = out.std()
            mean_out = out.mean()
            print('after ReLU->mean:%.3f\t std_mean:%.3f\t std_min:%.3f\t std_max:%.3f\t std:%.3f'%(mean_out.item(),std_out.mean().item(),std_out.min().item(),std_out.max().item(),std.item()))
        '''
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10, float=False):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = first_conv(3, 16, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], float=float)
        self.layer2 = self._make_layer(
            block, 32, layers[1], stride=2, float=float)
        self.layer3 = self._make_layer(
            block, 64, layers[2], stride=2, float=float)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        #self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = last_fc(64 * block.expansion, num_classes)

        self.init_error_dict()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, float=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QuantConv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False)
                if float is False else nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                                                 stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                      stride, downsample, float=float))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, float=float))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d):
                m.show_params()

    def init_error_dict(self):
        self.ew_dict = OrderedDict()
        self.ex_dict = OrderedDict()
        for k, m in self.named_modules():
            if isinstance(m, QuantConv2d):
                self.ew_dict[k] = torch.zeros(m.out_channels).cuda()
                self.ex_dict[k] = torch.zeros(m.in_channels).cuda()

    def cmp_sensitivity(self):
        for k, m in self.named_modules():
            if isinstance(m, QuantConv2d):
                error_w, error_x = m.cmp_sensitivity()
                self.ew_dict[k] += error_w.abs()
                self.ex_dict[k] += error_x.abs()

    def avg_bitwidth(self):
        '''
        compute total avarage bitwidth to terminate decreasing kernel-wise bitwidth
        '''
        total_w = 0.
        base_w = 0.
        total_a = 0.
        base_a = 0.
        for k, m in self.named_modules():
            if isinstance(m, QuantConv2d):
                total_w += m.bit_W.data.sum().mul(m.k_size).mul(m.k_size).mul(m.in_channels).item()
                base_w += m.k_size * m.k_size * m.out_channels * m.in_channels
                total_a += m.bit_A.data.sum().mul(
                    m.x_q.shape[2]).mul(m.x_q.shape[3]).item()
                base_a += m.x_q.shape[1] * m.x_q.shape[2] * m.x_q.shape[3]

        avg_bw = total_w / base_w
        avg_ba = total_a / base_a
        return avg_bw, avg_ba

    def adjust_bw(self, obj_bw=2.0, obj_bx=2.0, w_ratio=0.1, x_ratio=0.1):
        '''
        adjust kernel-wise bitwidth based on  computed sensitivity 
        '''
        avg_bw, avg_bx = self.avg_bitwidth()
        is_dw = avg_bw > obj_bw
        is_dx = avg_bx > obj_bx and avg_bx < 16

        if is_dw or is_dx:

            ew_list = None
            ex_list = None
            kw_num = 0
            kw_mark = [kw_num]
            kx_num = 0
            kx_mark = [kx_num]
            for k, m in self.named_modules():
                if isinstance(m, QuantConv2d):
                    if ew_list is None or ex_list is None:
                        ew_list = self.ew_dict[k].abs().cpu().numpy()
                        ex_list = self.ex_dict[k].abs().cpu().numpy()

                    else:
                        ew_list = np.append(
                            ew_list, self.ew_dict[k].abs().cpu().numpy())
                        ex_list = np.append(
                            ex_list, self.ex_dict[k].abs().cpu().numpy())
                    kw_num += m.out_channels
                    kw_mark.append(kw_num)
                    kx_num += m.in_channels
                    kx_mark.append(kx_num)

            # the number of weight kernels to decrease bitwidth
            dw_num = int(kw_num * w_ratio)
            # the number of feature channels to decrease bitwidth
            dx_num = int(kx_num * x_ratio)

            w_args = np.argsort(ew_list)
            x_args = np.argsort(ex_list)

            dw_list = np.zeros(ew_list.shape)
            dx_list = np.zeros(ex_list.shape)
            dw_list[w_args[0:dw_num]] = 1
            dx_list[x_args[0:dx_num]] = 1

            i = 0
            for k, m in self.named_modules():
                if isinstance(m, QuantConv2d):
                    if is_dw:
                        dw_m = dw_list[kw_mark[i]:kw_mark[i+1]]

                        w_new = m.bit_W.data.cpu().numpy() - dw_m
                        w_new = np.clip(w_new, 0, 8)

                    else:
                        w_new = None
                    if is_dx:
                        x_new = m.bit_A.data.cpu().numpy() - \
                            dx_list[kx_mark[i]:kx_mark[i+1]]
                        x_new = np.clip(x_new, 1, 8)
                    else:
                        x_new = None
                    m.update_state(w_new, x_new)
                    # m.set_params_fn(w_new,None)
                    i += 1
            self.init_error_dict()
        return avg_bw, avg_bx


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    pass
    # net = resnet20_cifar(float=True)
    # y = net(torch.randn(1, 3, 64, 64))
    # print(net)
    # print(y.size())
