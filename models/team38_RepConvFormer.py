import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc
from itertools import repeat


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1, bias=True)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = self.lkb_origin.weight, self.lkb_origin.bias
        if hasattr(self, 'small_conv'):
            small_k, small_b = self.small_conv.weight, self.small_conv.bias
            with torch.no_grad():
                eq_b += small_b
                #   add to the central part
                eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv2d(in_channels=self.lkb_origin.in_channels,
                                     out_channels=self.lkb_origin.out_channels,
                                     kernel_size=self.lkb_origin.kernel_size, stride=self.lkb_origin.stride,
                                     padding=self.lkb_origin.padding, dilation=self.lkb_origin.dilation,
                                     groups=self.lkb_origin.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x

    
class ConvMod(nn.Module):
    def __init__(self, dim, dw_size, small_kernel, small_kernel_merged=False):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                ReparamLargeKernelConv(dim, dim, dw_size, stride=1, groups=dim, 
                                       small_kernel=small_kernel, small_kernel_merged=small_kernel_merged)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        
        x = self.norm(x)   
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)

        return x

class Block(nn.Module):
    def __init__(self, dim, dw_size, small_kernel, small_kernel_merged=False, mlp_ratio=4.):
        super().__init__()
        
        self.attn = ConvMod(dim, dw_size, small_kernel, small_kernel_merged)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6           
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x)
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)
        return x
    
class Layer(nn.Module):
    def __init__(self, dim, depth, dw_size, small_kernel, small_kernel_merged=False, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, dw_size, small_kernel, small_kernel_merged, mlp_ratio) for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.conv(x) + x
    
class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops
    
class RepConvFormer(nn.Module):
    def __init__(self, img_size=64, in_chans=3, embed_dim=96, depths=(6, 6, 6, 6), dw_size=3, small_kernel=7, 
                 mlp_ratio=4., small_kernel_merged=False, upscale=2, img_range=1., upsampler=''):
        super(RepConvFormer, self).__init__()
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.num_layers = len(depths)
        
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Layer(embed_dim, depths[i_layer], dw_size, small_kernel, small_kernel_merged, mlp_ratio)
            self.layers.append(layer)
        self.norm = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        if self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, in_chans)
        else:
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return x

    def forward(self, x):
        h, w = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x
