#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
borrowed from https://github.com/facebookresearch/mae/blob/main/models_vit.py
"""
from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer


def LinearActivation(x):
    return x


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "linear": LinearActivation}   


from collections import OrderedDict
class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class LoreftIntervention(torch.nn.Module):
    """
    LoReFT(h) = h + R^T(Wh + b − Rh)
    """
    def __init__(self, **kwargs):
        super().__init__()
        if "embed_dim" in kwargs and kwargs["embed_dim"] is not None:
            self.register_buffer('embed_dim', torch.tensor(kwargs["embed_dim"]))
            self.register_buffer('interchange_dim', torch.tensor(kwargs["embed_dim"]))
        else:
            self.embed_dim = None
            self.interchange_dim = None

        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(self.embed_dim, kwargs["low_rank_dimension"])
        self.dropout = torch.nn.Dropout(kwargs["dropout"] if "dropout" in kwargs else 0.0)
        self.act_fn = ACT2FN["linear"] if "activation" not in kwargs or kwargs["activation"] is None else ACT2FN[kwargs["activation"]]
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base), self.rotate_layer.weight.T
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        self.learned_source.load_state_dict(state_dict, strict=False)
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base[:,:overload_w_width] = overload_w
        return
    
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


import torch_dct as dct
class Block(nn.Module):

    def __init__(self, reft_cfg, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, prompt_cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if prompt_cfg is not None:
            self.num_tokens = prompt_cfg.NUM_TOKENS
        else:
            self.num_tokens = 0

        self.reft = LoreftIntervention(
            embed_dim=dim, low_rank_dimension=reft_cfg.RANK,
            dropout=reft_cfg.DROPOUT, activation=reft_cfg.ACTIVATION
        )
        nn.init.xavier_uniform_(self.reft.learned_source.weight)
        nn.init.normal_(self.reft.learned_source.bias, std=1e-6)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        ### apply one intervention to all patch tokens
        patch_token = x[:,(1+self.num_tokens):,:]
        dct_patch_token = dct.dct(patch_token)
        dct_intervened_patch_token = self.reft(dct_patch_token)
        idct_intervened_patch_token = dct.idct(dct_intervened_patch_token)
        intervened_x = torch.cat((x[:, 0:(1+self.num_tokens), :], idct_intervened_patch_token), dim=1)  

        return intervened_x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, reft_cfg, prompt_cfg=None, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        dpr = [x.item() for x in torch.linspace(0, 0.1, 12)]  # stochastic depth decay rule 
        self.blocks = nn.Sequential(*[
            Block(
                reft_cfg, 
                drop_path=dpr[i], 
                dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                prompt_cfg=prompt_cfg)
            for i in range(12)])
        
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def build_model(model_type):
    if "vitb" in model_type:
        return vit_base_patch16()
    elif "vitl" in model_type:
        return vit_large_patch16()
    elif "vith" in model_type:
        return vit_huge_patch14()


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        drop_path_rate=0.1, global_pool=True,  # using default settings for mae-finetune
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
