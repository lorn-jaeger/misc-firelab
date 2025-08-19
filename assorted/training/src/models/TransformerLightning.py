from typing import Any

import torch
from .BaseModel import BaseModel
from .TransformerModel import TransformerModel

class TransformerLightning(BaseModel):
    """
    U-Net architecture with a pre-trained transformer model in the bottleneck and skip connections.
    """
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        transformer_model_path: str,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            use_doy=True,  # Transformer model uses the day of the year as an input feature
            *args,
            **kwargs
        )

        self.transformer_model = TransformerModel(
            input_dim=n_channels,
            num_layers=6,
            num_heads=8,
            dim_feedforward=2048,
            dropout=0.1,
            pre_trained_path=transformer_model_path
        )

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        return self.transformer_model(x, doys)