import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import matplotlib
matplotlib.use("Agg")   # Must be before importing pyplot
import matplotlib.pyplot as plt
import torch.nn as nn
from timm.layers import DropPath
from torchvision.models import convnext_small
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import torch
from torch import nn
import torch.fft
import random as random

##----------------------------------------------------------------------------------------
class EdgeAttention(nn.Module):
    def __init__(self, in_channel, dropout_prob=0.5):
        super(EdgeAttention, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU(in_channel)
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x):
        edge = x - self.avg_pool(x)
        weight = self.sigmoid(self.bn1(self.conv_1(edge)))
        out = weight * x + x
        out = self.prelu(out)
        out = self.dropout(out)
        return out

##----------------------------------------------------------------------------------------


##----------------------------------------------------------------------------------------
class GlobalLocalFeature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalLocalFeature, self).__init__()
        self.dilLayer=nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, stride=1,dilation=5, padding=5)
        self.bn1=nn.BatchNorm2d(in_channels//2)
        self.act1=nn.GELU()
        self.con1=nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, stride=1, padding=1)
        self.bn2=nn.BatchNorm2d(in_channels//2)
        self.act2=nn.GELU()
        self.forConcat=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
    def forward(self, x):
        origFea = x
        x1 = self.dilLayer(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x2 = self.con1(x)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)
        combineFeat =torch.concatenate([x1, x2], 1)
        out = self.forConcat(combineFeat)
        out = out+origFea
        return out

##----------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

##----------------------------------------------------------------------------------------
class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=10,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )

            if i==0:
                stage.add_module('CustomBlock1', GlobalLocalFeature(in_channels=96, out_channels=96))
            if i==1:
                stage.add_module('CustomBlock2', GlobalLocalFeature(in_channels=192, out_channels=192))
            if i==2:
                stage.add_module('CustomBlock3', GlobalLocalFeature(in_channels=384, out_channels=384))
            if i==3:
                stage.add_module('CustomBlock4',GlobalLocalFeature(in_channels=768,out_channels=768))
                stage.add_module('CustomBlock5', EdgeAttention(in_channel=768))



            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



model = ConvNeXt(depths=[3,3,27,3], dims=[96,192,384,768])