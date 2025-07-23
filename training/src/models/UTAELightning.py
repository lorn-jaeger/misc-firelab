from typing import Any

import torch
import os
from .BaseModel import BaseModel
from .utae_paps_models.utae import UTAE


class UTAELightning(BaseModel):
    """_summary_ U-Net architecture with temporal attention in the bottleneck and skip connections.
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
            use_doy=True, # UTAE uses the day of the year as an input feature
            *args,
            **kwargs
        )

        self.model = UTAE(
            input_dim=n_channels,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 1],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode="att_group",
            encoder_norm="group",
            n_head=16,
            d_model=256,
            d_k=4,
            encoder=False,
            return_maps=False,
            pad_value=0,
            padding_mode="reflect",
        )
        encoder_weights = encoder_weights if encoder_weights != "none" else None
        if encoder_weights == "pastis": 
            primary_ckpt = '/develop/data/utae_pre/model.pth.tar'
            secondary_ckpt = '/home/sl221120/WildfireSpreadTS/src/models/utae_paps_models/model.pth.tar'
            pretrained_checkpoint = primary_ckpt if os.path.exists(primary_ckpt) else secondary_ckpt
            self.load_checkpoint(pretrained_checkpoint)


    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a pretrained checkpoint for the model.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["state_dict"]
        prefix = "encoder."
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_state_dict[key[len(prefix):]] = value
            else:
                new_state_dict[key] = value
        model_state_dict = self.model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in new_state_dict.items()
            if k in model_state_dict and model_state_dict[k].size() == v.size()
        }
        # Load the weights into the model
        self.model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Checkpoint loaded successfully from '{checkpoint_path}'")

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        return self.model(x, batch_positions=doys, return_att=False)
