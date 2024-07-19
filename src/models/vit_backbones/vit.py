#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
models for vits, borrowed from
https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling_resnet.py
https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
"""
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from ...configs import vit_configs as configs
logger = logging.getLogger(__name__)


CONFIGS = {
    "sup_vitb16_224": configs.get_b16_config(),
    "sup_vitb16": configs.get_b16_config(),
    "sup_vitl16_224": configs.get_l16_config(),
    "sup_vitl16": configs.get_l16_config(),
    "sup_vitb16_imagenet21k": configs.get_b16_config(),
    "sup_vitl16_imagenet21k": configs.get_l16_config(),
    "sup_vitl32_imagenet21k": configs.get_l32_config(),
    'sup_vitb32_imagenet21k': configs.get_b32_config(),
    'sup_vitb8_imagenet21k': configs.get_b8_config(),
    'sup_vith14_imagenet21k': configs.get_h14_config(),
}


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


def LinearActivation(x):
    return x
    

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish, "linear": LinearActivation}   


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
    

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) # B, num_patches, head_size*num_head
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) # B, num_head, num_patches, head_size
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer) # B, num_head, num_patches, head_size

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # B, num_head, num_patches, num_patches
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores) # B, num_head, num_patches(query), num_patches(key)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # B, num_head, num_patches, head_size
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h 

        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # pjoin -> '/'.join((ROOT, ATTENTION_Q, "kernel")) for Windows
            query_weight = np2th(weights['/'.join((ROOT, ATTENTION_Q, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights['/'.join((ROOT, ATTENTION_K, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights['/'.join((ROOT, ATTENTION_V, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights['/'.join((ROOT, ATTENTION_OUT, "kernel"))]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights['/'.join((ROOT, ATTENTION_Q, "bias"))]).view(-1)
            key_bias = np2th(weights['/'.join((ROOT, ATTENTION_K, "bias"))]).view(-1)
            value_bias = np2th(weights['/'.join((ROOT, ATTENTION_V, "bias"))]).view(-1)
            out_bias = np2th(weights['/'.join((ROOT, ATTENTION_OUT, "bias"))]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights['/'.join((ROOT, FC_0, "kernel"))]).t()
            mlp_weight_1 = np2th(weights['/'.join((ROOT, FC_1, "kernel"))]).t()
            mlp_bias_0 = np2th(weights['/'.join((ROOT, FC_0, "bias"))]).t()
            mlp_bias_1 = np2th(weights['/'.join((ROOT, FC_1, "bias"))]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights['/'.join((ROOT, ATTENTION_NORM, "scale"))]))
            self.attention_norm.bias.copy_(np2th(weights['/'.join((ROOT, ATTENTION_NORM, "bias"))]))
            self.ffn_norm.weight.copy_(np2th(weights['/'.join((ROOT, MLP_NORM, "scale"))]))
            self.ffn_norm.bias.copy_(np2th(weights['/'.join((ROOT, MLP_NORM, "bias"))]))


################################################################################################  
class Block_REFT_single(nn.Module):
    def __init__(self, config, reft_cfg, vis):
        super(Block_REFT_single, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

        self.reft = LoreftIntervention(
            embed_dim=config.hidden_size, low_rank_dimension=reft_cfg.RANK,
            dropout=reft_cfg.DROPOUT, activation=reft_cfg.ACTIVATION
        )
        nn.init.xavier_uniform_(self.reft.learned_source.weight)
        nn.init.normal_(self.reft.learned_source.bias, std=1e-6)


    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h # x ([batch_size, 197, 768])

        # ### add intervention for CLS tokens
        # cls_token = x[:,0,:]
        # intervened_cls_token = self.reft(cls_token)
        # intervened_x = torch.cat((intervened_cls_token.unsqueeze(1), x[:, 1:, :]), dim=1)

        ### apply one intervention to all patch tokens
        patch_token = x[:,1:,:]
        intervened_patch_token = self.reft(patch_token)
        intervened_x = torch.cat((x[:, 0, :].unsqueeze(1), intervened_patch_token), dim=1)   

        return intervened_x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # pjoin -> '/'.join((ROOT, ATTENTION_Q, "kernel")) for Windows
            query_weight = np2th(weights['/'.join((ROOT, ATTENTION_Q, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights['/'.join((ROOT, ATTENTION_K, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights['/'.join((ROOT, ATTENTION_V, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights['/'.join((ROOT, ATTENTION_OUT, "kernel"))]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights['/'.join((ROOT, ATTENTION_Q, "bias"))]).view(-1)
            key_bias = np2th(weights['/'.join((ROOT, ATTENTION_K, "bias"))]).view(-1)
            value_bias = np2th(weights['/'.join((ROOT, ATTENTION_V, "bias"))]).view(-1)
            out_bias = np2th(weights['/'.join((ROOT, ATTENTION_OUT, "bias"))]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights['/'.join((ROOT, FC_0, "kernel"))]).t()
            mlp_weight_1 = np2th(weights['/'.join((ROOT, FC_1, "kernel"))]).t()
            mlp_bias_0 = np2th(weights['/'.join((ROOT, FC_0, "bias"))]).t()
            mlp_bias_1 = np2th(weights['/'.join((ROOT, FC_1, "bias"))]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights['/'.join((ROOT, ATTENTION_NORM, "scale"))]))
            self.attention_norm.bias.copy_(np2th(weights['/'.join((ROOT, ATTENTION_NORM, "bias"))]))
            self.ffn_norm.weight.copy_(np2th(weights['/'.join((ROOT, MLP_NORM, "scale"))]))
            self.ffn_norm.bias.copy_(np2th(weights['/'.join((ROOT, MLP_NORM, "bias"))]))


import torch_dct as dct
class Block_REFT_single_dct(nn.Module):
    def __init__(self, config, reft_cfg, vis, prompt_cfg=None):
        super(Block_REFT_single_dct, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)
        
        if prompt_cfg is not None:
            self.num_tokens = prompt_cfg.NUM_TOKENS
        else:
            self.num_tokens = 0
            
        self.reft = LoreftIntervention(
            embed_dim=config.hidden_size, low_rank_dimension=reft_cfg.RANK,
            dropout=reft_cfg.DROPOUT, activation=reft_cfg.ACTIVATION
        )
        nn.init.xavier_uniform_(self.reft.learned_source.weight)
        nn.init.normal_(self.reft.learned_source.bias, std=1e-6)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h # x ([batch_size, 197, 768])

        ### apply one intervention to all patch tokens
        patch_token = x[:,(1+self.num_tokens):,:]
        dct_patch_token = dct.dct(patch_token)
        dct_intervened_patch_token = self.reft(dct_patch_token)
        idct_intervened_patch_token = dct.idct(dct_intervened_patch_token)
        intervened_x = torch.cat((x[:, 0:(1+self.num_tokens), :], idct_intervened_patch_token), dim=1)  

        ### apply one intervention to selected patch tokens
        # index = np.array(range(196))
        # index = index.reshape((14,14))
        # selected = index[3:11,3:11]
        # selected = selected.reshape(-1)

        # selected = np.random.randint(0, 196, 50)
        
        # patch_token = x[:,selected+11,:]
        # dct_patch_token = dct.dct(patch_token)
        # dct_intervened_patch_token = self.reft(dct_patch_token)
        # idct_intervened_patch_token = dct.idct(dct_intervened_patch_token)
 
        # ### apply one intervention to each patch token
        # reft_outputs = []
        # j=0
        # for i in range(x.shape[1]-1):
        #     if i not in selected:
        #         reft_outputs.append(x[:,i+1,:].unsqueeze(1))
        #     else:
        #         reft_outputs.append(idct_intervened_patch_token[:,j,:].unsqueeze(1))
        #         j += 1
        # intervened_patch_tokens = torch.cat(reft_outputs, dim=1)
        # intervened_x = torch.cat((x[:, 0, :].unsqueeze(1), intervened_patch_tokens), dim=1)

        return intervened_x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # pjoin -> '/'.join((ROOT, ATTENTION_Q, "kernel")) for Windows
            query_weight = np2th(weights['/'.join((ROOT, ATTENTION_Q, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights['/'.join((ROOT, ATTENTION_K, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights['/'.join((ROOT, ATTENTION_V, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights['/'.join((ROOT, ATTENTION_OUT, "kernel"))]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights['/'.join((ROOT, ATTENTION_Q, "bias"))]).view(-1)
            key_bias = np2th(weights['/'.join((ROOT, ATTENTION_K, "bias"))]).view(-1)
            value_bias = np2th(weights['/'.join((ROOT, ATTENTION_V, "bias"))]).view(-1)
            out_bias = np2th(weights['/'.join((ROOT, ATTENTION_OUT, "bias"))]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights['/'.join((ROOT, FC_0, "kernel"))]).t()
            mlp_weight_1 = np2th(weights['/'.join((ROOT, FC_1, "kernel"))]).t()
            mlp_bias_0 = np2th(weights['/'.join((ROOT, FC_0, "bias"))]).t()
            mlp_bias_1 = np2th(weights['/'.join((ROOT, FC_1, "bias"))]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights['/'.join((ROOT, ATTENTION_NORM, "scale"))]))
            self.attention_norm.bias.copy_(np2th(weights['/'.join((ROOT, ATTENTION_NORM, "bias"))]))
            self.ffn_norm.weight.copy_(np2th(weights['/'.join((ROOT, MLP_NORM, "scale"))]))
            self.ffn_norm.bias.copy_(np2th(weights['/'.join((ROOT, MLP_NORM, "bias"))]))


class Block_REFT_double(nn.Module):
    def __init__(self, config, reft_cfg, vis):
        super(Block_REFT_double, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

        self.reft_cls = LoreftIntervention(
            embed_dim=config.hidden_size, low_rank_dimension=reft_cfg.RANK,
            dropout=reft_cfg.DROPOUT, activation=reft_cfg.ACTIVATION
        )
        self.reft_image = LoreftIntervention(
            embed_dim=config.hidden_size, low_rank_dimension=reft_cfg.RANK,
            dropout=reft_cfg.DROPOUT, activation=reft_cfg.ACTIVATION
        )

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h # x ([batch_size, 197, 768])

        ### add intervention for CLS tokens
        cls_token = x[:,0,:]
        intervened_cls_token = self.reft_cls(cls_token)

        # ### apply one intervention to all patch tokens
        # patch_token = x[:,1:,:]
        # intervened_patch_token = self.reft_image(patch_token)

        # intervened_x = torch.cat((intervened_cls_token.unsqueeze(1), intervened_patch_token), dim=1)

        ### apply one intervention to selected patch tokens
        index = np.array(range(196))
        index = index.reshape((14,14))
        selected = index[3:11,3:11]
        selected = selected.reshape(-1)

        patch_token = x[:,selected+1,:]
        dct_patch_token = dct.dct(patch_token)
        dct_intervened_patch_token = self.reft(dct_patch_token)
        idct_intervened_patch_token = dct.idct(dct_intervened_patch_token)
 
        ### apply one intervention to each patch token
        reft_outputs = []
        j=0
        for i in range(x.shape[1]-1):
            if i not in selected:
                reft_outputs.append(x[:,i+1,:].unsqueeze(1))
            else:
                reft_outputs.append(idct_intervened_patch_token[:,j,:].unsqueeze(1))
                j += 1
        intervened_patch_tokens = torch.cat(reft_outputs, dim=1)
        intervened_x = torch.cat((intervened_cls_token.unsqueeze(1), intervened_patch_tokens), dim=1)  

        return intervened_x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            # pjoin -> '/'.join((ROOT, ATTENTION_Q, "kernel")) for Windows
            query_weight = np2th(weights['/'.join((ROOT, ATTENTION_Q, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights['/'.join((ROOT, ATTENTION_K, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights['/'.join((ROOT, ATTENTION_V, "kernel"))]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights['/'.join((ROOT, ATTENTION_OUT, "kernel"))]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights['/'.join((ROOT, ATTENTION_Q, "bias"))]).view(-1)
            key_bias = np2th(weights['/'.join((ROOT, ATTENTION_K, "bias"))]).view(-1)
            value_bias = np2th(weights['/'.join((ROOT, ATTENTION_V, "bias"))]).view(-1)
            out_bias = np2th(weights['/'.join((ROOT, ATTENTION_OUT, "bias"))]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights['/'.join((ROOT, FC_0, "kernel"))]).t()
            mlp_weight_1 = np2th(weights['/'.join((ROOT, FC_1, "kernel"))]).t()
            mlp_bias_0 = np2th(weights['/'.join((ROOT, FC_0, "bias"))]).t()
            mlp_bias_1 = np2th(weights['/'.join((ROOT, FC_1, "bias"))]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights['/'.join((ROOT, ATTENTION_NORM, "scale"))]))
            self.attention_norm.bias.copy_(np2th(weights['/'.join((ROOT, ATTENTION_NORM, "bias"))]))
            self.ffn_norm.weight.copy_(np2th(weights['/'.join((ROOT, MLP_NORM, "scale"))]))
            self.ffn_norm.bias.copy_(np2th(weights['/'.join((ROOT, MLP_NORM, "bias"))]))


class Encoder(nn.Module):
    def __init__(self, config, reft_cfg, vis, prompt_cfg=None):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        if reft_cfg.ALLLAYERS:
            layers = list(range(config.transformer["num_layers"]))
        else:
            layers = reft_cfg.LAYERS
        for layer_index in range(config.transformer["num_layers"]):
            if layer_index in layers:
                if reft_cfg.DOUBLE:
                    layer = Block_REFT_double(config, reft_cfg, vis)
                    self.layer.append(copy.deepcopy(layer))
                elif reft_cfg.DCT:
                    layer = Block_REFT_single_dct(config, reft_cfg, vis, prompt_cfg)
                    self.layer.append(copy.deepcopy(layer))
                else:
                    layer = Block_REFT_single(config, reft_cfg, vis)
                    self.layer.append(copy.deepcopy(layer))
            else:
                layer = Block(config, vis)
                self.layer.append(copy.deepcopy(layer))
 
    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)  
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward_cls_layerwise(self, hidden_states):
        # hidden_states: B, 1+n_patches, dim

        if hidden_states.size(0) != 1:
            raise ValueError('not support batch-wise cls forward yet')
        
        cls_embeds = []
        cls_embeds.append(hidden_states[0][0])
        for i,layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if i < len(self.layer)-1:
                cls_embeds.append(hidden_states[0][0])
        encoded = self.encoder_norm(hidden_states)
        cls_embeds.append(hidden_states[0][0])

        cls_embeds = torch.stack(cls_embeds) # 12, dim
        return cls_embeds


class Transformer(nn.Module):
    def __init__(self, config, reft_cfg, img_size, vis, prompt_cfg=None): 
        super(Transformer, self).__init__()

        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, reft_cfg, vis, prompt_cfg) 

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights
    
    def forward_cls_layerwise(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        cls_embeds = self.encoder.forward_cls_layerwise(embedding_output)
        return cls_embeds


class VisionTransformer(nn.Module):
    def __init__(
        self, model_type, reft_cfg,
        img_size=224, num_classes=21843, vis=False
    ):
        super(VisionTransformer, self).__init__()
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier
        
        self.transformer = Transformer(config, reft_cfg, img_size, vis) 
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights # attn_weights: num_layers, B, num_head, num_patches, num_patches
    
    def forward_cls_layerwise(self, x):
        cls_embeds = self.transformer.forward_cls_layerwise(x)
        return cls_embeds

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

