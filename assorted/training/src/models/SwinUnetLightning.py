from typing import Any

import torch
import os
from .BaseModel import BaseModel
from .SwinUnet.networks.vision_transformer import SwinUnet


class SwinUnetLightning(BaseModel):
    """_summary_ SwinUnet architecture with skip connections.
    """
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        encoder_weights = None,
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
        
        class Config:
            class DATA:
                IMG_SIZE = 224 
            
            class MODEL:
                DROP_RATE = 0.0  
                DROP_PATH_RATE = 0.2
                LABEL_SMOOTHING = 0.1
                NAME = 'swin_tiny_patch4_window7_224'
                if encoder_weights == "imagenet":
                    primary_ckpt = '/develop/data/swin_tiny_patch4_window7_224.pth'
                    secondary_ckpt = '/home/sl221120/develop/data/swin_tiny_patch4_window7_224.pth'
                    PRETRAIN_CKPT = primary_ckpt if os.path.exists(primary_ckpt) else secondary_ckpt
                else:
                    PRETRAIN_CKPT = None
                
                class SWIN:
                    PATCH_SIZE = 4 
                    IN_CHANS = n_channels  
                    EMBED_DIM = 96  
                    DEPTHS = [2, 2, 2, 2]  
                    NUM_HEADS = [3, 6, 12, 24] 
                    WINDOW_SIZE = 7
                    MLP_RATIO = 4.0  
                    QKV_BIAS = True  
                    QK_SCALE = None 
                    APE = False
                    PATCH_NORM = True

            class TRAIN:
                USE_CHECKPOINT = False
            
        config = Config()

        self.model = SwinUnet(config, num_classes=1)
        self.model.load_from(config)
