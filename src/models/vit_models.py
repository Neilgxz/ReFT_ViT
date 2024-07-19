#!/usr/bin/env python3
"""
ViT-related models
Note: models return logits instead of prob
"""
import torch.nn as nn
from .build_vit_backbone import (
    build_vit_sup_models, build_swin_model,
    build_mocov3_model, build_mae_model
)
from .mlp import MLP
from ..utils import logging
logger = logging.get_logger("visual_reft")


class ViT(nn.Module):
    """ViT-related model."""

    def __init__(self, cfg, load_pretrain=True, vis=False):
        super(ViT, self).__init__()

        if "reft" in cfg.MODEL.TRANSFER_TYPE:
            reft_cfg = cfg.MODEL.REFT
        else:
            reft_cfg = None

        if "prompt" in cfg.MODEL.TRANSFER_TYPE:
            prompt_cfg = cfg.MODEL.PROMPT
        else:
            prompt_cfg = None

        if cfg.MODEL.TRANSFER_TYPE != "end2end" and "reft" not in cfg.MODEL.TRANSFER_TYPE:
            self.froze_enc = True
        else:
            self.froze_enc = False

        self.build_backbone(cfg, reft_cfg, prompt_cfg, load_pretrain, vis=vis)
        self.cfg = cfg
        self.setup_head(cfg)

    def build_backbone(self, cfg, reft_cfg, prompt_cfg, load_pretrain, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_vit_sup_models(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, reft_cfg, prompt_cfg, cfg.MODEL.MODEL_ROOT, load_pretrain, vis
        )

        if transfer_type == "linear" or transfer_type == "reft":
            for k, p in self.enc.named_parameters():
                if 'reft' not in k: 
                    p.requires_grad = False
        elif transfer_type == "reft_prompt":   
            for k, p in self.enc.named_parameters():
                if 'reft' not in k and "prompt" not in k: 
                    p.requires_grad = False               
        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k: 
                    p.requires_grad = False
        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))

    def setup_head(self, cfg):
        self.head = MLP(
            input_dim=self.feat_dim,
            mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )

    def forward(self, x, return_feature=False):

        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim

        if return_feature:
            return x, x
        x = self.head(x)

        return x
    
    def forward_cls_layerwise(self, x):
        cls_embeds = self.enc.forward_cls_layerwise(x)
        return cls_embeds

    def get_features(self, x):
        """get a (batch_size, self.feat_dim) feature"""
        x = self.enc(x)  # batch_size x self.feat_dim
        return x


class SSLViT(ViT):
    """moco-v3 and mae model."""

    def __init__(self, cfg):
        super(SSLViT, self).__init__(cfg)

    def build_backbone(self, cfg, reft_cfg, prompt_cfg, load_pretrain, vis):
        if "moco" in cfg.DATA.FEATURE:
            build_fn = build_mocov3_model
        elif "mae" in cfg.DATA.FEATURE:
            build_fn = build_mae_model

        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_fn(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, reft_cfg, prompt_cfg, cfg.MODEL.MODEL_ROOT
        )

        if transfer_type == "linear" or transfer_type == "reft":
            for k, p in self.enc.named_parameters():
                if 'reft' not in k: 
                    p.requires_grad = False
        elif transfer_type == "reft_prompt":   
            for k, p in self.enc.named_parameters():
                if 'reft' not in k and "prompt" not in k: 
                    p.requires_grad = False               
        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k: 
                    p.requires_grad = False
        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")
        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))


class Swin(ViT):
    """Swin-related model."""

    def __init__(self, cfg):
        super(Swin, self).__init__(cfg)

    def build_backbone(self, prompt_cfg, cfg, adapter_cfg, load_pretrain, vis):
        transfer_type = cfg.MODEL.TRANSFER_TYPE
        self.enc, self.feat_dim = build_swin_model(
            cfg.DATA.FEATURE, cfg.DATA.CROPSIZE,
            prompt_cfg, cfg.MODEL.MODEL_ROOT
        )

        # linear, prompt, cls, cls+prompt, partial_1
        if transfer_type == "partial-1":
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-1].blocks)
            for k, p in self.enc.named_parameters():
                if "layers.{}.blocks.{}".format(total_layer - 1, total_blocks - 1) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-2":
            total_layer = len(self.enc.layers)
            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "partial-4":
            total_layer = len(self.enc.layers)
            total_blocks = len(self.enc.layers[-2].blocks)

            for k, p in self.enc.named_parameters():
                if "layers.{}".format(total_layer - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2, total_blocks - 1) not in k and "layers.{}.blocks.{}".format(total_layer - 2, total_blocks - 2) not in k and "layers.{}.downsample".format(total_layer - 2) not in k and "norm.weight" != k and "norm.bias" != k: # noqa
                    p.requires_grad = False

        elif transfer_type == "linear" or transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt" and prompt_cfg.LOCATION in ["below"]:
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "patch_embed" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif transfer_type == "end2end":
            logger.info("Enable all parameters update during training")

        else:
            raise ValueError("transfer type {} is not supported".format(
                transfer_type))
