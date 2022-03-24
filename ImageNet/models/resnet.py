import os
import math
import torch
import torch.nn as nn
import torchvision.models
import numpy as np
from torch.utils.model_zoo import load_url
from .quant_layer import QuantConv2d,first_conv,last_fc
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# you need to download the models to ~/.torch/models
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
 }
models_dir = os.path.expanduser('~/.torch/models')
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}

'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
'''
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QuantConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QuantConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = first_conv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = last_fc(512 * block.expansion, num_classes)

        self.init_error_dict()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,
                          stride=stride,),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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
        for k,m in self.named_modules():
            if isinstance(m,QuantConv2d):
                self.ew_dict[k] = torch.zeros(m.out_channels).cuda()
                self.ex_dict[k] = torch.zeros(m.in_channels).cuda()
    def cmp_sensitivity(self):
        for k,m in self.named_modules():
            if isinstance(m,QuantConv2d):
                error_w,error_x= m.cmp_sensitivity()
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
        for k,m in self.named_modules():
            if isinstance(m,QuantConv2d):
                total_w += m.bit_W.data.sum().mul(m.k_size).mul(m.k_size).mul(m.in_channels).item()
                base_w  += m.k_size * m.k_size * m.out_channels * m.in_channels
                total_a += m.bit_A.data.sum().mul(m.x_q.shape[2]).mul(m.x_q.shape[3]).item()
                base_a += m.x_q.shape[1] * m.x_q.shape[2] * m.x_q.shape[3]
        
        avg_bw = total_w / base_w
        avg_ba = total_a / base_a
        return avg_bw,avg_ba
    def adjust_bw(self,obj_bw=2.0,obj_bx=2.0,w_ratio=0.1,x_ratio=0.1):
        '''
        adjust kernel-wise bitwidth based on  computed sensitivity 
        '''
        avg_bw,avg_bx = self.avg_bitwidth()
        is_dw = avg_bw > obj_bw
        is_dx = avg_bx > obj_bx and avg_bx < 16

        if is_dw or is_dx:

            ew_list = None
            ex_list = None
            kw_num = 0
            kw_mark = [kw_num]
            kx_num = 0
            kx_mark = [kx_num]
            for k,m in self.named_modules():
                if isinstance(m,QuantConv2d):
                    if  ew_list is None or ex_list is None:
                        ew_list = self.ew_dict[k].abs().cpu().numpy()
                        ex_list = self.ex_dict[k].abs().cpu().numpy()
                        
                    else:
                        ew_list = np.append(ew_list,self.ew_dict[k].abs().cpu().numpy())
                        ex_list = np.append(ex_list,self.ex_dict[k].abs().cpu().numpy())
                    kw_num += m.out_channels
                    kw_mark.append(kw_num)
                    kx_num += m.in_channels
                    kx_mark.append(kx_num)    
                
            dw_num = int(kw_num * w_ratio)     # the number of weight kernels to decrease bitwidth 
            dx_num = int(kx_num * x_ratio)     # the number of feature channels to decrease bitwidth

            w_args = np.argsort(ew_list)
            x_args = np.argsort(ex_list)

            dw_list = np.zeros(ew_list.shape)
            dx_list = np.zeros(ex_list.shape)
            dw_list[w_args[0:dw_num]] = 1
            dx_list[x_args[0:dx_num]] = 1

            i = 0 
            for k,m in self.named_modules():
                if isinstance(m,QuantConv2d):
                    if is_dw:
                        dw_m = dw_list[kw_mark[i]:kw_mark[i+1]]

                        w_new = m.bit_W.data.cpu().numpy() - dw_m
                        w_new = np.clip(w_new,0,8)

                    else:
                        w_new = None
                    if is_dx:
                        x_new = m.bit_A.data.cpu().numpy() - dx_list[kx_mark[i]:kx_mark[i+1]]
                        x_new = np.clip(x_new,1,8)
                    else:
                        x_new = None
                    m.update_state(w_new,x_new)
                    #m.set_params_fn(w_new,None)
                    i += 1   
            self.init_error_dict()
        return avg_bw,avg_bx

    


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = load_url(model_urls['resnet18'],models_dir, progress=True)
        model.load_state_dict(state_dict, strict=False)
        #model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet18'])), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_url(model_urls['resnet34'],models_dir, progress=True)
        model.load_state_dict(state_dict, strict=False)
        #model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet34'])), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = load_url(model_urls['resnet50'],models_dir, progress=True)
        model.load_state_dict(state_dict, strict=False)
       #model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet50'])), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet101'])))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name['resnet152'])))
    return model
