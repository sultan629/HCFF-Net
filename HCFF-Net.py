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
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

##----------------------------------------------------------------------------------------
class EMA_LightFreq(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA_LightFreq, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv_dw = DepthwiseSeparableConv(channels // self.groups, channels // self.groups)
        self.freq_proj = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1)
    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv_dw(group_x)
        freq = torch.fft.rfft2(group_x, norm="ortho")
        freq_mag = torch.abs(freq)  # magnitude spectrum
        freq_mag = torch.nn.functional.interpolate(freq_mag.unsqueeze(1).mean(-1), size=(h, w), mode="bilinear")
        freq_feat = self.freq_proj(freq_mag.repeat(1, group_x.size(1), 1, 1))
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        weights = weights + freq_feat
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

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
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

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
#
#
model_urls = {
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth"
}




@register_model
def convnext_small(pretrained=True, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

model = ConvNeXt(depths=[3,3,27,3], dims=[96,192,384,768])   #### this is DRC-Net

saved_model_path = 'D:\\TrainedModels\\CustomModel2\\Train_Val_CustomModel_07_Sep_ACID_.pt'
model.load_state_dict(torch.load(saved_model_path),strict=False)
model.head = nn.Linear(768, 10)
model.to(device)
print("✅ ConvNeXt backbone loaded, head initialized for 10 classes")
print(model)

densenet = models.densenet121(pretrained=True)
densenet.classifier = nn.Linear(densenet.classifier.in_features, 10)
densenet_ckpt = torch.load("D:\\TrainedModels\\DenseNet121\\Train_Val_result_14_Apr_ACID_denseNet-121_.pt", map_location="cuda")
densenet.load_state_dict(densenet_ckpt, strict=False)
print("✅ Loaded your trained DenseNet121 weights.")


class SpatialConcatFusion(nn.Module):
    def __init__(self, convnext, densenet, num_classes=10):
        super(SpatialConcatFusion, self).__init__()
        self.convnext = convnext
        self.densenet = densenet

        self.classifier = nn.Linear(768 + 1024, num_classes)

    def extract_convnext_features(self, x):
        # Run through ConvNeXt until last stage (no pooling)
        for i in range(4):
            x = self.convnext.downsample_layers[i](x)
            x = self.convnext.stages[i](x)
        return x  # (N, 768, 7, 7)

    def forward(self, x):
        # ---- ConvNeXt feature map ----
        with torch.no_grad():
            feat_convnext = self.extract_convnext_features(x)  # (N, 768, 7, 7)

        # ---- DenseNet feature map ----
        with torch.no_grad():
            feat_densenet = self.densenet.features(x)
            feat_densenet = F.relu(feat_densenet)  # (N, 1024, 7, 7)

        fused = torch.cat((feat_convnext, feat_densenet), dim=1)  # (N, 1792, 7, 7)
        fused = F.adaptive_avg_pool2d(fused, (1, 1)).view(fused.size(0), -1)  # (N, 1792)
        out = self.classifier(fused)
        return out

finalModel=SpatialConcatFusion(model, densenet)

saved_model_path = r"D:\TrainedModels\26 Feb HCFF_Net\weights_HCFF-Net.pt"
finalModel.load_state_dict(torch.load(saved_model_path))
finalModel.to(device)
print(finalModel)
