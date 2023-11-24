# -*- codeing = utf-8 -*-
# @Time : 2023/10/26 12:19
# @Author : 李昌杏
# @File : student.py
# @Software : PyCharm
import math
import warnings
from collections import OrderedDict

import timm
import torch
import torch.nn as nn
from timm.models import VisionTransformer
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class Student_IMG(nn.Module):
    def __init__(self, num_classes, feature_dim=768,representation=256,encoder_backbone='vit_base_patch16_224',checkpoint_path='../../weights/vit.npz'):
        super().__init__()

        self.num_classes = num_classes
        self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=False,
                                                            checkpoint_path=checkpoint_path)
        self.encoder.embed_dim=feature_dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes)
        )
        self.encoder.embed_dim=feature_dim
        self.fc = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim,representation)
        )

    def embedding(self, photo):
        x = self.encoder.patch_embed(photo)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x[:,0]

    def forward_features(self, photo):
        return self.encoder.forward_features(photo)

    def classify(self, features):
        return self.mlp_head(features)

    def forward(self, photo):#class_logits,representation,cls_token,
        x=self.embedding(photo)
        return self.classify(x),self.fc(x),x
    def f(self,photo):
        x = self.encoder.patch_embed(photo)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x

class Student_SKT(nn.Module):
    def __init__(self, num_classes, feature_dim=768, representation=256, encoder_backbone='vit_base_patch16_224',
                 checkpoint_path='../../weights/vit.npz'):
        super().__init__()

        self.num_classes = num_classes
        self.encoder: VisionTransformer = timm.create_model(encoder_backbone, pretrained=False,
                                                            checkpoint_path=checkpoint_path)
        self.encoder.embed_dim=feature_dim
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes)
        )
        self.encoder.embed_dim=feature_dim
        self.fc = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, representation)
        )

        self.scale = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feature_dim, 3, 2, 1, bias=False),
        )

    def embedding(self, photo):
        x = self.encoder.patch_embed(photo)
        b, h_w, d = x.shape
        x1 = self.scale(photo).view(b, d, h_w).transpose(1, 2)
        x = (x + x1) / 2
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x[:, 0]

    def forward_features(self, photo):
        return self.encoder.forward_features(photo)

    def classify(self, features):
        return self.mlp_head(features)

    def forward(self, photo):  # class_logits,representation,cls_token,
        x = self.embedding(photo)
        return self.classify(x), self.fc(x), x

    def f(self,photo):
        x = self.encoder.patch_embed(photo)
        b, h_w, d = x.shape
        x1 = self.scale(photo).view(b, d, h_w).transpose(1, 2)
        x = (x + x1) / 2
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x
