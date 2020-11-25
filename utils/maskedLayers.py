# Taken from here
# https://github.com/wanglouis49/pytorch-weights_pruning/blob/master/pruning/layers.py

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LinearMasked(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearMasked, self).__init__(in_features, out_features, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = Variable(mask, requires_grad=False, volatile=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class Conv2dMasked(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv2dMasked, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = Variable(mask, requires_grad=False, volatile=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.conv2d(
                x,
                weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
