'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from models.quant_layer import *
from collections import OrderedDict
import numpy as np

__all__ = [
    'VGG', 'cifar10_vggsmall'
]

model_urls = {
    'vgg7': 'https://github.com/rhhc/zxd_releases/releases/download/Re/cifar10-vggsmall-zxd-93.4-8943fa3.pth',
}


def cifar10_vggsmall(pretrained=False, float=False, **kwargs):
    """VGG small model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    if float:
        print('float model!')
        model = VGG('VGG9')
    else:
        print('non float model!')
        model = VGG_Q('VGG9')
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg9']))
    return model


cfg = {
    'VGG9': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
}


class VGG_Q(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_Q, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(512 * 16, 10)

        self.classifier = nn.Sequential(
            QuantLinear(512*16, 1024, 8, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            QuantLinear(1024, 1024, 1, 512),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            last_fc(1024, 10)
        )
        '''
        self.classifier = nn.Sequential(
            #QuantLinear(512*16,1024,8,1024),
            #nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            #nn.ReLU(inplace=True),           
            #QuantLinear(1024,1024,1,512),
            #nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            #nn.ReLU(inplace=True),
            last_fc(512*16, 10)
        )
        '''
        self._initialize_weights()
        self.init_error_dict()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        first_layer = True
        for x in cfg:
            if first_layer:
                layers += [first_conv(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                first_layer = False
                continue
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [QuantConv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                m.show_params()

    def init_error_dict(self):
        self.ew_dict = OrderedDict()
        self.ex_dict = OrderedDict()
        for k, m in self.named_modules():
            if isinstance(m, QuantConv2d):
                self.ew_dict[k] = torch.zeros(m.out_channels).cuda()
                self.ex_dict[k] = torch.zeros(m.in_channels).cuda()
            if isinstance(m, QuantLinear):
                self.ew_dict[k] = torch.zeros(m.out_groups).cuda()
                self.ex_dict[k] = torch.zeros(m.in_groups).cuda()

    def cmp_sensitivity(self):
        for k, m in self.named_modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
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

            if isinstance(m, QuantLinear):
                total_w += m.bit_W.data.sum().mul(m.out_features * m.in_features / m.out_groups).item()
                base_w += m.out_features * m.in_features
                total_a += m.bit_A.data.sum().mul(m.in_features / m.in_groups).item()
                base_a += m.in_features

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
                if isinstance(m, QuantLinear):
                    if ew_list is None or ex_list is None:
                        ew_list = self.ew_dict[k].abs().cpu().numpy()
                        ex_list = self.ex_dict[k].abs().cpu().numpy()

                    else:
                        ew_list = np.append(
                            ew_list, self.ew_dict[k].abs().cpu().numpy())
                        ex_list = np.append(
                            ex_list, self.ex_dict[k].abs().cpu().numpy())
                    kw_num += m.out_groups
                    kw_mark.append(kw_num)
                    kx_num += m.in_groups
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
                if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
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


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        #self.classifier = nn.Linear(512 * 16, 10)
        self.classifier = nn.Sequential(
            nn.Linear(512*16, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(num_features=1024, affine=True, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )
        '''
        self.classifier = nn.Sequential(
            nn.Linear(512*16, 10)
        )
        '''
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        first_layer = True
        for x in cfg:
            if first_layer:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                first_layer = False
                continue
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
