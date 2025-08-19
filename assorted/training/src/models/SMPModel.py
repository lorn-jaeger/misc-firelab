from typing import Any

import segmentation_models_pytorch as smp

from .BaseModel import BaseModel


class SMPModel(BaseModel):
    """_summary_ Segmentation model based on the SMP package. We only use the U-Net model. 
    """
    def __init__(
        self,
        encoder_name: str,
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
            *args,
            **kwargs
        )
        self.save_hyperparameters()
        encoder_weights = encoder_weights if encoder_weights != "none" else None
        self.model = smp.Unet(
            encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=n_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
        )
        print(f"Loaded {encoder_name} with {encoder_weights} weights")

        
