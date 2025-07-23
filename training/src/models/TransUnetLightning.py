from typing import Any

import numpy as np
import os
from .BaseModel import BaseModel
from .TransUnet.networks.vit_seg_modeling import VisionTransformer as TransUnet
from .TransUnet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


class TransUnetLightning(BaseModel):
    """_summary_ TransUnet architecture.
    """
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        encoder_weights: str = None,  # e.g. "imagenet" or "none"
        vit_name: str = "R50-ViT-B_16",
        vit_patches_size: int = 16,
        n_skip: int = 3,
        img_size: int = 224,
        *args: Any,
        **kwargs: Any

    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            use_doy=False, 
            *args,
            **kwargs
        )
        encoder_weights = encoder_weights if encoder_weights != "none" else None

        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = 1  
        config_vit.n_skip = n_skip
        primary_ckpt = '/develop/data/imagenet21k_R50+ViT-B_16.npz'
        secondary_ckpt = '/home/sl221120/WildfireSpreadTS/src/models/TransUnet/imagenet21k_R50+ViT-B_16.npz'
        config_vit.pretrained_path = primary_ckpt if os.path.exists(primary_ckpt) else secondary_ckpt

        if "R50" in vit_name:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

        self.model = TransUnet(config_vit, img_size=img_size, num_classes=config_vit.n_classes, in_chans=n_channels)

        if encoder_weights == "imagenet":
            weights = np.load(config_vit.pretrained_path, allow_pickle=True)
            self.model.load_from(weights)
