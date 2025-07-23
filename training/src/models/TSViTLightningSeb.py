from typing import Any, Tuple
import torch

from .BaseModel import BaseModel
from .TSViT_models.TSViTdense_Saad import TSViT

class TSViTLightningSeb(BaseModel):

    def __init__(
            self, 
            n_channels: int, 
            flatten_temporal_dimension: bool, 
            pos_class_weight: float, 
            img_size:Tuple[int, int], 
            depth: int = 2, 
            *args: Any, 
            **kwargs: Any):
        super().__init__(
            n_channels=n_channels, 
            flatten_temporal_dimension=flatten_temporal_dimension, 
            pos_class_weight=pos_class_weight, 
            use_doy=True, 
            required_img_size=img_size, 
            *args, 
            **kwargs)
        model_config = {
            "img_res": 128, 
            "max_seq_len": 5, 
            "num_channels": n_channels+1, 
            "num_classes": 1, 
            "ignore_background": False, 
            "dropout": 0., 
            "patch_size": 2, 
            "dim": 128, 
            "temporal_depth": depth, # number of blocks in temporal transformer,
            "spatial_depth": depth, # number of blocks in spatial transformer
            "heads": 4,
            "pool": 'cls', 
            "dim_head": 32, 
            "emb_dropout": 0., 
            "scale_dim": 4}

        self.model = TSViT(model_config)

    def forward(self, x: torch.Tensor, doys:torch.Tensor) -> torch.Tensor:
        # The transformer model expects a day of year feature as the last channel to compute a temporal positional embedding.
        # B, T, C, H, W = x.shape
        # # 0-indexing, instead of 1-indexing. Then divide by 365 to normalize in the way that the model expects it.
        # doys -= 1
        # xt = doys.reshape(B, T, 1, 1, 1).repeat(1, 1, 1, H, W)/365.0
        # # Add day of year as last channel
        # x = torch.cat([x, xt], dim=2)
        return self.model(x, doys)