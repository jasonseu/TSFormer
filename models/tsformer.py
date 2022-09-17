# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-3-22
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import math
import torch
import torch.nn as nn
import numpy as np

from .timm_models.vision_transformer import *
from .timm_models.util.layers import DropPath, Mlp
from .factory import register_model


@register_model
class TSFormer(nn.Module):
    def __init__(self, pretrained, cfg, embed_dim=768, depth=12):
        super().__init__()
        self.img_encoder = vit_base_patch16_224_in21k(pretrained=pretrained, img_size=cfg.img_size)
        if cfg.embed_type == 'bert' or cfg.embed_type == 'glove':
            self.text_feats = torch.tensor(np.load(cfg.embed_path), dtype=torch.float32).cuda()
        elif cfg.embed_type == 'random':
            self.text_feats = torch.eye(cfg.num_classes).cuda()
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=cfg.num_heads) for _ in range(depth)])
        self.cls_head = nn.Linear(embed_dim, cfg.num_classes)
        self.patch_head = nn.Linear(embed_dim, cfg.num_classes)
        self.text_head = TextHead(embed_dim, cfg.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.cfg = cfg
        self.text_linear = nn.Linear(cfg.num_classes, embed_dim, bias=False)
        self.depth = depth
        
    def forward(self, x):
        batch_size = x.size(0)
        vision_feats = self.img_encoder(x)
        tfeat = torch.stack([self.text_feats for _ in range(batch_size)], dim=0)
        if self.cfg.embed_type == 'random':
            tfeat = self.text_linear(tfeat)
        for i in range(self.cfg.start_depth, self.depth):
            vfeat = vision_feats[i]
            tfeat, attn = self.blocks[i](vfeat, tfeat)
        
        logits = self.text_head(tfeat)
        
        if self.training:
            return logits
        
        return logits, attn[..., 1:]


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.vnorm1 = norm_layer(dim)
        self.tnorm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, vfeat, tfeat):
        x, attn = self.attn(self.vnorm1(vfeat), self.tnorm1(tfeat))
        tfeat = tfeat + self.drop_path(x)
        tfeat = tfeat + self.drop_path(self.mlp(self.norm2(tfeat)))
        return tfeat, attn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, vfeat, tfeat):
        B, Nv, C = vfeat.shape
        Nt = tfeat.size(1)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q = self.query(tfeat).reshape(B, Nt, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(vfeat).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(vfeat).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, Nt, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class TextHead(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
                    
    def forward(self, x):
        x = torch.sum(x * self.weight, 2)
        if self.bias is not None:
            x = x + self.bias
        return x