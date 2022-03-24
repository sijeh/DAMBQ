import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

ALPHA_W_0BIT = [0.0,0.0,0.0,0.0]
ALPHA_W_1BIT = [1.0,0.0,0.0,0.0]
ALPHA_W_2BIT = [0.6123,0.3877,0.0,0.0]
ALPHA_W_3BIT = [0.4557,0.3436,0.2007,0.0]
ALPHA_W_4BIT = [0.3317,0.2858,0.2346,0.1479]
WEIGHT_BASE = [-1,1]
CLIP_FACTOR_W = torch.tensor([0.5,0.98,2.6,4.16,5.66])

def build_quant_base():
    quant_base = torch.zeros(5,16)
    count = 0
    for i in WEIGHT_BASE:
        for j in WEIGHT_BASE:
            for k in WEIGHT_BASE:
                for l in WEIGHT_BASE:
                    quant_base[0,count] = ALPHA_W_0BIT[0] * i + ALPHA_W_0BIT[1] * j +  ALPHA_W_0BIT[2] * k +  ALPHA_W_0BIT[3] * l 
                    quant_base[1,count] = ALPHA_W_1BIT[0] * i + ALPHA_W_1BIT[1] * j +  ALPHA_W_1BIT[2] * k +  ALPHA_W_1BIT[3] * l
                    quant_base[2,count] = ALPHA_W_2BIT[0] * i + ALPHA_W_2BIT[1] * j +  ALPHA_W_2BIT[2] * k +  ALPHA_W_2BIT[3] * l
                    quant_base[3,count] = ALPHA_W_3BIT[0] * i + ALPHA_W_3BIT[1] * j +  ALPHA_W_3BIT[2] * k +  ALPHA_W_3BIT[3] * l
                    quant_base[4,count] = ALPHA_W_4BIT[0] * i + ALPHA_W_4BIT[1] * j +  ALPHA_W_4BIT[2] * k +  ALPHA_W_4BIT[3] * l
                    count += 1
    return quant_base

QUANT_BASE = build_quant_base()

def build_proj_set(bit_width):
    return QUANT_BASE.to(bit_width.device).index_select(0,bit_width)

def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

def quantize_wgt(data,clip_scale,proj_set):
    x = data / clip_scale
    x = torch.clamp(x,-1,1)
    xshape = x.shape
    pshape = proj_set.shape
    xhard = x.view(xshape[0],1,-1)
    phard = proj_set.view(pshape[0],-1,1)
    idxs = (xhard - phard).abs().min(dim=1)[1]
    xhard = torch.gather(input=proj_set,dim=1,index=idxs).view(xshape)
    y = (xhard - x).detach() + x
    y = y * clip_scale
    return y

def quantize_act(data,clip_scale,bitwidth):
    quant_scale = 2**bitwidth - 1
    x = data / clip_scale
    x = torch.clamp(x,0,1) * quant_scale
    y = round_pass(x)
    y = y * clip_scale / quant_scale 
    return y

class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups,bias)
        self.layer_type = 'QuantConv2d'
        self.k_size = kernel_size
        #self.bit_W = nbits
        #self.bit_A = nbits
        #self.alpha_W = Parameter(torch.Tensor(1))
        #self.alpha_A = Parameter(torch.Tensor(1))
        #self.alpha_W = Parameter(torch.full((self.out_channels,1,1,1),1.0))
        #self.alpha_A = Parameter(torch.full((1,self.in_channels,1,1),3.0))
        #self.alpha_W = Parameter(torch.tensor(1.0))
        #self.alpha_A = Parameter(torch.tensor(3.0))
        #self.register_buffer('is_init', torch.zeros(1))
        #self.register_buffer('alpha_A',torch.full((1,self.in_channels,1,1),1.0))
        #self.register_buffer('alpha_A_biased',torch.full((1,self.in_channels,1,1),1.0)) 
        self.register_buffer('alpha_A',torch.tensor(2.0))
        #self.register_buffer('alpha_A_biased',torch.tensor(3.0))          
        self.register_buffer('alpha_W',torch.full((self.out_channels,1,1,1),1.0))
        #self.register_buffer('alpha_W_biased',torch.full((self.out_channels,1,1,1),1.0))
        ema_decay=0.99
        self.register_buffer('ema_decay',torch.tensor(ema_decay))
        #self.register_buffer('iter_count', torch.zeros(1))
        

    def forward(self,x):
        '''
        if self.bit_A < 0 or self.bit_W < 0:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        
        if self.training and self.is_init == 0:
            #self.alpha_W.data.copy_(self.weight.abs().max() / 2 ** (self.bit_W - 1))
            #self.alpha_A.data.copy_(x.abs().max() / 2 ** (self.bit_A - 1))
            self.alpha_W.data.copy_(self.weight.abs().max())
            self.alpha_A.data.copy_(x.abs().max())

            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.is_init.fill_(1)
        '''
        #wgt_mean = self.weight.data.mean(dim=(1,2,3),keepdim=True)
        #wgt_std = self.weight.data.std(dim=(1,2,3),keepdim=True)
        #wgt_mean = self.weight.data.mean()
        #wgt_std = self.weight.data.std()
        #norm_weight = self.weight.add(-wgt_mean).div(wgt_std)   #weights normalization
        #norm_weight = self.weight
        
        if self.training:
            with torch.no_grad():
                b_W = self.weight.abs().mean(dim=(1,2,3),keepdim=True)
                #b_A = 2.0 * x.abs().mean(dim=(0,2,3),keepdim=True)
                alpha_W_cur = CLIP_FACTOR_W[self.bit_W.view(self.out_channels,1,1,1)].to(self.bit_W.device) * b_W
                #alpha_A_cur = CLIP_FACTOR_A[self.bit_A.view(1,self.in_channels,1,1)].to(self.bit_A.device) * b_A
                #alpha_A_cur = 6.2 * b_A

                #alpha_A_cur = x.mean(dim=(0,2,3),keepdim=True) + 3 * x.std(dim=(0,2,3),keepdim=True)
                #alpha_A_cur = x.mean() + 3 * x.std()
            #self.iter_count += 1
            #self.alpha_W_biased.data,self.alpha_W.data = update_ema(self.alpha_W_biased,alpha_W_cur,
            #                                                        self.ema_decay,self.iter_count)
            #self.alpha_A_biased.data,self.alpha_A.data = update_ema(self.alpha_A_biased,alpha_A_cur,
            #                                                        self.ema_decay,self.iter_count)
            self.alpha_W.data = alpha_W_cur * (1 - self.ema_decay) + self.ema_decay * self.alpha_W.data
            #self.alpha_A.data = alpha_A_cur * (1 - self.ema_decay) + self.ema_decay * self.alpha_A.data
            #print(self.alpha_A.data)

        if self.bit_A_init == 32:
            self.x_q = x
        else:
            self.x_q = quantize_act(x,self.alpha_A,self.bit_A.view(1,self.in_channels,1,1))
        self.x_res = x - self.x_q
        if self.training:
            self.x_q.retain_grad()
        self.w_q = quantize_wgt(self.weight,self.alpha_W,self.value_Q)
        self.w_res = self.weight - self.w_q
        if self.training:
            self.w_q.retain_grad()
        return F.conv2d(self.x_q, self.w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
    def init_state(self,bit_W_init=None,bit_A_init=None):
        if bit_W_init is not None:
            assert isinstance(bit_W_init,int)
            self.bit_W_init = bit_W_init
            bit_W = [bit_W_init for i in range(self.out_channels)]
            self.register_buffer('bit_W',torch.tensor(bit_W))
            quant_value = build_proj_set(self.bit_W)
            self.register_buffer('value_Q',quant_value)

            #self.quant_wgt = quant_wgt_fn(self.bit_W.data.view(-1,1,1,1))
            
        if bit_A_init is not None:
            assert isinstance(bit_A_init,int)
            self.bit_A_init = bit_A_init
            bit_A = [bit_A_init for i in range(self.in_channels)]
            self.register_buffer('bit_A',torch.tensor(bit_A))
    
    def update_state(self,bit_W=None,bit_A=None):
        if bit_W is not None:
            if isinstance(bit_W,int):
                bit_W = [bit_W for i in range(self.out_channels)]
            elif isinstance(bit_W,list):
                assert len(bit_W) == self.out_channels
                bit_W = [int(i) for i in bit_W]
            elif isinstance(bit_W,np.ndarray):
                assert len(bit_W) == self.out_channels
                bit_W = bit_W.astype(int)
                #print('bit_W.shape: ',bit_W.shape)
            else:
                raise TypeError('bit_W must be int or list or np.ndarray!')
            
            self.bit_W.data = torch.tensor(bit_W).to(self.bit_W.device)
            quant_value = build_proj_set(self.bit_W)
            self.value_Q.data = torch.tensor(quant_value).to(self.bit_W.device)

        if bit_A is not None:
            if isinstance(bit_A,int):
                bit_A = [bit_A for i in range(self.in_channels)]
            elif isinstance(bit_W,list) :
                assert len(bit_A) == self.in_channels
                bit_A = [int(i) for i in bit_A]
            elif isinstance(bit_A,np.ndarray):
                assert len(bit_A) == self.in_channels
                bit_A = bit_A.astype(int)
                #print('bit_A.shape: ',bit_A.shape)
            else:
                raise TypeError('bit_A must be int or list or np.ndarray!')

            self.bit_A.data = torch.tensor(bit_A).to(self.bit_A.device)
    
    def cmp_sensitivity(self):
        '''
        compute kernel-wise sensitivity to decide to increase or decrease kernel-wise bitwidth
        '''
        #compute first order error of self.w_q
        def first_order_error():

            #w_num = float(self.w_res.nelement())
            w_num = float(self.w_res.nelement()/self.out_channels)
            error_w = self.w_res.div(w_num).mul(self.w_q.grad).sum(dim=(1,2,3),keepdim=False).detach()
            
            #x_num = float(self.x_res.nelement())
            x_num = float(self.x_res.nelement()/self.in_channels)
            error_x = self.x_res.div(x_num).mul(self.x_q.grad).sum(dim=(0,2,3),keepdim=False).detach()

            return error_w,error_x    
        
        error_w,error_x = first_order_error()  
        return error_w,error_x

    def show_params(self):
        alpha_W = round(self.alpha_W.data.mean().item(), 3)
        alpha_A = round(self.alpha_A.data.mean().item(), 3)
        print('alpha_W: {:2f}, alpha_A: {:2f}'.format(alpha_W, alpha_A))
    def avg_bitwidth(self):
        avg_bw = self.bit_W.data.type(torch.float).mean().item()
        avg_ba = self.bit_A.data.type(torch.float).mean().item()
        return avg_bw,avg_ba

class first_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(first_conv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.layer_type = 'FConv2d'
        bit_W = 8
        self.Qn_W = - 2**(bit_W-1) + 0.5
        self.Qp_W = 2**(bit_W-1) - 0.5

    def forward(self, x):
        max_W = self.weight.data.abs().max()
        w_q = ((self.weight/max_W*self.Qp_W - self.Qn_W).round() + self.Qn_W) / self.Qp_W * max_W
        w_q = (w_q - self.weight).detach() + self.weight
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class last_fc(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(last_fc, self).__init__(in_features, out_features, bias)
        self.layer_type = 'LFC'
        bit_W = 8
        self.Qn_W = - 2**(bit_W-1) + 0.5
        self.Qp_W = 2**(bit_W-1) - 0.5
        bit_A = 8 
        self.Qn_A = 0
        self.Qp_A = 2**bit_A - 1

    def forward(self, x):
        max_W = self.weight.data.abs().max()
        w_q = ((self.weight/max_W*self.Qp_W - self.Qn_W).round() + self.Qn_W) / self.Qp_W * max_W
        w_q = (w_q - self.weight).detach() + self.weight
        max_A = x.abs().max()
        x_q = ((x/max_A*self.Qp_A - self.Qn_A).round() + self.Qn_A) / self.Qp_A * max_A
        x_q = (x_q - x).detach() + x
        return F.linear(x_q, w_q, self.bias)

if __name__ == '__main__':
    quant_base = build_quant_base()
    print(quant_base)
    index = torch.tensor([4,3])
    out = build_proj_set(index)
    print(out)
    
    x = torch.tensor([[-0.1,-0.2,0.3,0.4,0.5,0.6],[-0.4,0.8,0.3,0.4,0.5,0.6]])
    y = quantize(x,0.5,out,)
    print(y)