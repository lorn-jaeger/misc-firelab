from typing import Any

import torch

from .BaseModel import BaseModel
from .TSViT_models.TSViTdense_Saad import TSViT

class TSViTLightning(BaseModel):
    """_summary_ Temporal-Spatial Vision Transformer for object classification.
    """
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        img_res: int,
        patch_size: int,
        max_seq_len: int,
        dim: int,
        temporal_depth: int,
        spatial_depth: int,
        heads: int,
        dim_head: int,
        dropout: float,
        emb_dropout: float,
        scale_dim: int,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            use_doy=True,  # TSViT uses the day of the year as an input feature
            *args,
            **kwargs
        )

        model_config = {
            'img_res': img_res,
            'patch_size': patch_size,
            'num_classes': 1,  # Assuming binary classification
            'max_seq_len': max_seq_len,
            'dim': dim,
            'temporal_depth': temporal_depth,
            'spatial_depth': spatial_depth,
            'heads': heads,
            'dim_head': dim_head,
            'dropout': dropout,
            'emb_dropout': emb_dropout,
            'pool': 'cls',
            'num_channels': n_channels,
            'scale_dim': scale_dim
        }

        self.model = TSViT(model_config)

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        # TSViT expects the day of year to be the last channel of the input
        #doys = doys.unsqueeze(-1).unsqueeze(-1)  # Now [B, T, 1, 1]
        # Step 2: Expand doys to match the height and width of x
        #doys = doys.expand(-1, -1, x.shape[-2], x.shape[-1])  # Now [B, T, H, W]
        # Step 3: Concatenate doys to x along the channel dimension
        #x = torch.cat([x, doys.unsqueeze(2)], dim=2)  # Now [B, T, C+1, H, W]
        # Pass the concatenated tensor through the model
        #print("doy inside lightning", doys)
        #print("doy shape inside lightning", doys.shape)
        return self.model(x, doys)